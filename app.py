"""
Flask API for IP-Composer multi-concept fusion.

Wraps demo_multi_concept.py's algorithm with HTTP endpoints, so a client can
upload a base image plus a set of (image, concept) pairs and get the fused
output directly — no YAML authoring step.

Endpoints
---------
  GET  /health            liveness probe + device info
  GET  /concepts          list built-in concepts (name, embeddings_path, rank)
  POST /compose           multi-concept fusion → generated PNG(s)
  GET  /outputs/<file>    serve a generated PNG

POST /compose request (multipart/form-data)
-------------------------------------------
  base_image            : file (required)
  slot_<i>              : file (required) — concept image for slot i (0-indexed)
  params                : form field, JSON string. Example:

    {
      "slots": [
        {"image_key": "slot_0", "concept": "age",      "alpha": 1.0,  "name": "age"},
        {"image_key": "slot_1", "concept": "emotions", "alpha": 0.8,  "name": "emotion"},
        {"image_key": "slot_2",
         "concept": {"text_variants": ["warm golden glow", "amber tone", ...]},
         "alpha": 0.6, "rank": 20, "name": "warmth"}
      ],
      "prompt": null,
      "scale": 1.0,
      "seed": 420,
      "num_samples": 4,
      "num_inference_steps": 50
    }

  `concept` is either:
    - str  : a built-in concept name from /concepts (no LLM call), OR a path
             to a .npy file, OR any free-form name — unknown names are auto-
             generated via LLM (gpt-5.4-2026-03-05 by default) and cached on
             disk under text_embeddings/ for reuse.
    - dict : {"text_variants": [...]} → encoded on-the-fly via CLIP text tower.
  `rank` and `alpha` are optional (defaults: 30, 1.0).

Environment (loaded from .env at startup):
  API_KEY            OpenAI-compatible API key (required for LLM auto-gen)
  API_BASE_URL       e.g. https://api.nuwaflux.com/v1
  LLM_MODEL          default: gpt-5.4-2026-03-05
  LLM_NUM_VARIANTS   default: 100
  LLM_TIMEOUT_S      default: 120

Run
---
  pip install flask
  python app.py
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (avoids python-dotenv dependency).

    Lines like  KEY=VALUE  are set into os.environ if not already present.
    Inline `#` comments are stripped. Quotes around values are preserved
    only if balanced. Does nothing if the file is absent.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.split("#", 1)[0].strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

import httpx
import numpy as np
import torch
import open_clip
from PIL import Image
from flask import Flask, abort, jsonify, request, send_from_directory
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from IP_Adapter import IPAdapterXL

from demo_multi_concept import compose_embedding, compute_projection_matrix


# ============================================================================
# Built-in concept registry (mirrors demo.py)
#   name → (npy_filename_under_text_embeddings, recommended_rank)
# ============================================================================
TEXT_EMBED_DIR = "text_embeddings"

CONCEPTS: Dict[str, Tuple[str, int]] = {
    "age":                              ("age_descriptions.npy",                 30),
    "animal fur":                       ("fur_descriptions.npy",                 80),
    "dogs":                             ("dog_descriptions.npy",                 30),
    "emotions":                         ("emotion_descriptions.npy",             30),
    "flowers":                          ("flower_descriptions.npy",              30),
    "fruit/vegetable":                  ("fruit_vegetable_descriptions.npy",     30),
    "outfit type":                      ("outfit_descriptions.npy",              30),
    "outfit pattern (including color)": ("outfit_pattern_descriptions.npy",      80),
    "outfit color":                     ("outfit_color_descriptions.npy",        30),
    "patterns":                         ("pattern_descriptions.npy",             80),
    "patterns (including color)":       ("pattern_descriptions_with_colors.npy", 80),
    "vehicle":                          ("vehicle_descriptions.npy",             30),
    "vehicle (including color)":        ("vehicle_descriptions_with_color.npy",  30),
    "daytime":                          ("times_of_day_descriptions.npy",        30),
    "pose":                             ("person_poses_descriptions.npy",        30),
    "season":                           ("season_descriptions.npy",              30),
    "material":                         ("material_descriptions.npy",            80),
}


# ============================================================================
# Auto-generated concept pipeline:
#   user-provided free-form name → LLM generates ~100 descriptions
#   → CLIP-text-encoded → saved as .npy → cached for future requests
# ============================================================================
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-2026-03-05")
LLM_NUM_VARIANTS = int(os.environ.get("LLM_NUM_VARIANTS", "100"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "120"))
DEFAULT_AUTO_RANK = 30

AUTO_CSV_DIR = "text_datasets"
AUTO_NPY_DIR = TEXT_EMBED_DIR  # reuse the same dir as built-ins
os.makedirs(AUTO_CSV_DIR, exist_ok=True)
os.makedirs(AUTO_NPY_DIR, exist_ok=True)

# Per-concept locks: prevent two concurrent requests from racing to LLM-generate
# the same new concept twice.
_CONCEPT_LOCKS: Dict[str, threading.Lock] = {}
_CONCEPT_LOCKS_GUARD = threading.Lock()


def _slug(name: str) -> str:
    """Normalize concept name into a filesystem-safe slug."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "concept"


def _pil_to_data_url(pil_img: "Image.Image", max_side: int = 512, quality: int = 85) -> str:
    """Encode a PIL image as a base64 JPEG data URL for vision LLM input.

    Downscales so the longer side ≤ max_side to keep token cost low (vision
    models don't need full resolution for axis disambiguation).
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _concept_lock(slug: str) -> threading.Lock:
    with _CONCEPT_LOCKS_GUARD:
        lk = _CONCEPT_LOCKS.get(slug)
        if lk is None:
            lk = threading.Lock()
            _CONCEPT_LOCKS[slug] = lk
        return lk


def _llm_generate_descriptions(
    concept_name: str,
    n: int,
    image_data_url: Optional[str] = None,
) -> List[str]:
    """Call an OpenAI-compatible chat API to produce `n` templated description
    sentences that span ONE visual axis of `concept_name`.

    Critical for IP-Composer correctness: every returned sentence shares an
    identical fixed scaffolding and differs only in a short "{X}" slot, so
    that SVD on the CLIP embeddings isolates a single concept axis rather
    than a mix of unrelated dimensions (color + finish + texture + ...).
    The LLM is forced to:
      1. Pick ONE visual axis implied by the concept name.
      2. Design a fixed sentence template with one variable slot.
      3. Generate N short variants that fill the slot.
    The server assembles `template.replace("{X}", variant)` per row.

    Reads API_BASE_URL / API_KEY from environment (loaded from .env at startup).
    """
    api_key = os.environ.get("API_KEY")
    api_base = os.environ.get("API_BASE_URL")
    if not api_key or not api_base:
        raise RuntimeError(
            "Auto-generation requires API_KEY and API_BASE_URL in environment "
            "(set them in .env)."
        )

    system_msg = (
        "You design 'concept probes' for the IP-Composer image-fusion algorithm. "
        "A probe is N templated sentences whose CLIP embeddings + SVD must yield "
        "a subspace that captures ONE coherent family of visual variation. "
        "Two competing principles you must balance: "
        "(A) AXIS PURITY — every sentence shares identical scaffolding, only {X} "
        "varies, so the subspace doesn't absorb scene/composition noise. "
        "(B) CONCEPT BREADTH — the {X} variants must span the FULL semantic range "
        "of the concept, distributed across all its major sub-aspects, not "
        "clustered inside one narrow sub-category. "
        "When a reference image is provided, use it to disambiguate the concept, "
        "but DO NOT over-fit variants to look like that one image — half the "
        "variants should describe instances visually DIFFERENT from the reference. "
        "Output strict JSON."
    )

    image_guidance = (
        "\n\nYou are ALSO given a REFERENCE IMAGE — the visual the user uploaded "
        "for this concept slot. The image is ONE specific instance along the axis "
        "you must identify. Use it to:\n"
        "  - Disambiguate the concept name. E.g. concept='outfit' + image of a red "
        "dress → axis is COLOR (variants: navy, emerald, mustard...), NOT garment "
        "type. Concept='outfit' + image of a tweed jacket → axis is FABRIC/TEXTURE "
        "(variants: linen, silk, leather, fleece...).\n"
        "  - Pick a template whose scaffolding is COMPATIBLE with the image's "
        "domain (don't write \"an outdoor scene with {X}\" if the image is an "
        "indoor object).\n"
        "  - BUT do NOT over-narrow: variants should include alternatives that "
        "look DIFFERENT from the reference, covering the whole axis range.\n"
    ) if image_data_url else ""

    user_msg = (
        f'Concept: "{concept_name}"'
        + image_guidance
        + "\n\nHARD REQUIREMENTS:\n"
        f"1. Identify the visual axis (or family of related axes) the concept "
        f"refers to. List its 3-8 major SUB-ASPECTS internally before generating "
        f"variants. E.g. \"glaze appearance\" has sub-aspects: color, sheen "
        f"(gloss/matte/satin), texture (smooth/crackle/speckle/runny), opacity.\n"
        f"2. Write ONE template with the placeholder \"{{X}}\". The scaffolding "
        f"words around {{X}} must appear IDENTICALLY in every output sentence.\n"
        f"3. Variants must NOT repeat any word that already appears in the "
        f"template scaffolding (e.g., if template is \"a ceramic surface coated "
        f"in a {{X}}\", variants must NOT contain 'ceramic', 'surface', 'coated', "
        f"'in', or 'a').\n"
        f"4. Generate exactly {n} variants that fill {{X}}, 1-10 words each. "
        f"Variants must be DISTINCT (no near-paraphrases like 'glossy red' and "
        f"'shiny red').\n"
        f"5. DISTRIBUTION RULE: spread the {n} variants approximately evenly "
        f"across ALL identified sub-aspects. Do NOT put more than ~30% of the "
        f"variants in any single sub-aspect. Variants MAY combine multiple "
        f"sub-aspects in one phrase (e.g., 'glossy cobalt blue with crackle' "
        f"combines color+sheen+texture).\n\n"
        f"Examples (note how variants SPAN sub-aspects):\n\n"
        f'  concept: "age of person"  (single discrete axis, short variants)\n'
        f'    axis: "human age stage"\n'
        f'    template: "A picture of a {{X}}"\n'
        f'    variants: ["newborn", "infant", "toddler", "young child", "preteen", '
        f'"teenager", "young adult", "middle-aged adult", "senior", "elderly person"]\n\n'
        f'  concept: "ceramic glaze finish"  (rich, multi-faceted: color×sheen×texture×opacity)\n'
        f'    axis: "glaze appearance"\n'
        f'    template: "a ceramic surface coated in a {{X}}"\n'
        f'    variants: ["glossy cobalt blue", "matte chalky white", '
        f'"crackled pale celadon", "speckled tan with iron flecks", '
        f'"translucent turquoise pooling", "metallic raku iridescence", '
        f'"smoky charcoal breaking brown", "honey amber crystalline", '
        f'"satin lavender", "blood-red high-gloss", ...] (note: every variant '
        f'mixes 1-3 sub-aspects; sub-aspects are evenly represented)\n\n'
        f'  concept: "surface pattern"  (single axis, varied pattern types)\n'
        f'    axis: "decorative pattern style"\n'
        f'    template: "an object with a {{X}} pattern"\n'
        f'    variants: ["striped", "polka-dot", "paisley", "plaid", '
        f'"checkered", "floral", "geometric", "zigzag", "mosaic", "lattice", ...]\n\n'
        f"Now process the concept above. Return JSON of exactly this shape:\n"
        f'{{"axis": "<one short phrase>", '
        f'"template": "<sentence with exactly one {{X}}>", '
        f'"variants": ["<v1>", "<v2>", ..., "<v{n}>"]}}'
    )

    # Build multimodal content for vision models when image_data_url is present;
    # fall back to plain text content otherwise.
    if image_data_url:
        user_content = [
            {"type": "text", "text": user_msg},
            {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}},
        ]
    else:
        user_content = user_msg

    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # trust_env=False: ignore HTTP(S)_PROXY env vars set by the user's shell
    # (e.g. clash-verge). The configured API_BASE_URL is expected to be directly
    # reachable; routing it through a foreign-traffic proxy breaks SSL.
    with httpx.Client(timeout=LLM_TIMEOUT_S, trust_env=False) as client:
        r = client.post(url, json=payload, headers=headers)

    # Some compatible endpoints don't support response_format; fall back to plain
    # text and try to extract JSON manually if needed.
    if r.status_code == 400 and "response_format" in r.text:
        payload.pop("response_format", None)
        with httpx.Client(timeout=LLM_TIMEOUT_S, trust_env=False) as client:
            r = client.post(url, json=payload, headers=headers)

    r.raise_for_status()
    body = r.json()
    content = body["choices"][0]["message"]["content"]

    # Try strict JSON first; if that fails, extract a {...} block.
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise RuntimeError(f"LLM did not return JSON; content head: {content[:200]!r}")
        parsed = json.loads(m.group(0))

    axis = str(parsed.get("axis", "")).strip()
    template = str(parsed.get("template", "")).strip()
    variants = parsed.get("variants") or []

    if "{X}" not in template:
        raise RuntimeError(
            f"LLM template missing required '{{X}}' slot. Got: {template!r}"
        )
    if not isinstance(variants, list):
        raise RuntimeError(f"LLM JSON missing 'variants' list; got: {parsed!r}")

    # Clean variants: strip, dedupe (case-insensitive), drop placeholder echoes,
    # and reject variants that repeat content words from the template scaffolding
    # (which would dilute the axis direction).
    template_words = re.findall(r"[a-zA-Z]+", template.replace("{X}", " "))
    STOPWORDS = {"a", "an", "the", "of", "in", "on", "with", "and", "or",
                 "is", "it", "to", "for", "by", "at"}
    blocked = {w.lower() for w in template_words if w.lower() not in STOPWORDS}

    seen = set()
    cleaned: List[str] = []
    overlap_drops = 0
    for v in variants:
        s = str(v).strip().lstrip("-•*").strip().strip('"').strip("'")
        key = s.lower()
        if not s or key in seen or "{x}" in key:
            continue
        # Drop if variant contains any non-stopword from the template
        v_words = {w.lower() for w in re.findall(r"[a-zA-Z]+", s)}
        if v_words & blocked:
            overlap_drops += 1
            continue
        seen.add(key)
        cleaned.append(s)
    if overlap_drops:
        print(f"[llm]   dropped {overlap_drops} variants that echoed template words "
              f"(blocked: {sorted(blocked)})")
    if len(cleaned) < 30:
        raise RuntimeError(
            f"LLM returned only {len(cleaned)} usable variants for "
            f"'{concept_name}' after dedup/overlap filtering; need ≥30. "
            f"Inspect log above for axis/template; consider rerunning."
        )

    sentences = [template.replace("{X}", v) for v in cleaned]
    print(f"[llm]   axis     : {axis}")
    print(f"[llm]   template : {template}")
    print(f"[llm]   variants : {len(cleaned)}  e.g. {cleaned[:3]}")
    return sentences


def _save_csv(path: str, phrases: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Description"])
        for p in phrases:
            w.writerow([p])


def ensure_concept_subspace(
    concept_name: str,
    runtime: "Runtime",
    slot_pil: Optional["Image.Image"] = None,
) -> Tuple[np.ndarray, int, str]:
    """Return (text_embeds, default_rank, source_descriptor) for a free-form
    concept name.

    Lookup order:
      1. text_embeddings/{slug}_descriptions.npy  (cached, either built-in or
         previously LLM-generated)
      2. Otherwise: call LLM (with slot_pil as a vision input if provided) →
         save CSV + NPY → load.

    Caching is by slug only — the slot image only influences axis selection
    on the FIRST call; later calls with the same concept name reuse the cached
    subspace regardless of which image is provided. To force regeneration,
    delete the .npy file under text_embeddings/.
    """
    slug = _slug(concept_name)
    npy_path = os.path.join(AUTO_NPY_DIR, f"{slug}_descriptions.npy")
    csv_path = os.path.join(AUTO_CSV_DIR, f"{slug}_descriptions.csv")

    with _concept_lock(slug):
        if os.path.exists(npy_path):
            text_embeds = np.load(npy_path)
            return text_embeds, DEFAULT_AUTO_RANK, f"cached:{slug}"

        data_url = _pil_to_data_url(slot_pil) if slot_pil is not None else None
        vision_tag = " with vision input" if data_url else ""
        print(f"[llm] generating {LLM_NUM_VARIANTS} descriptions for "
              f"concept='{concept_name}' (slug='{slug}') via {LLM_MODEL}{vision_tag} ...")
        t0 = time.time()
        phrases = _llm_generate_descriptions(
            concept_name, LLM_NUM_VARIANTS, image_data_url=data_url,
        )
        print(f"[llm] got {len(phrases)} phrases in {time.time() - t0:.1f}s; "
              f"encoding with CLIP ...")

        text_embeds = runtime.encode_texts(phrases)

        # Save both for reproducibility/inspection
        _save_csv(csv_path, phrases)
        np.save(npy_path, text_embeds)
        print(f"[llm] saved {csv_path} and {npy_path} (shape={text_embeds.shape})")

        return text_embeds, DEFAULT_AUTO_RANK, f"llm_generated:{slug}"


# ============================================================================
# Runtime: load CLIP + SDXL/IP-Adapter once at startup
# ============================================================================
class Runtime:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[startup] device={device}")

        print("[startup] loading CLIP-ViT-H-14 ...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        clip_model.to(device)
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        print("[startup] loading SDXL + IP-Adapter ...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        ip_ckpt = hf_hub_download(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            filename="ip-adapter_sdxl_vit-h.bin",
        )
        self.ip_model = IPAdapterXL(
            pipe, "h94/IP-Adapter", "models/image_encoder", ip_ckpt, device
        )

        self.lock = threading.Lock()  # serialize GPU work across requests
        print("[startup] ready")

    def encode_image_pil(self, pil_img: Image.Image) -> np.ndarray:
        x = self.preprocess(pil_img.convert("RGB"))[None].to(self.device)
        with torch.no_grad():
            e = self.clip_model.encode_image(x)
        return e.cpu().numpy()

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                e = self.clip_model.encode_text(tok)
            out.append(e.cpu().numpy())
        return np.vstack(out)


RUNTIME: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = Runtime()
    return RUNTIME


# ============================================================================
# Concept resolution: built-in name | .npy path | inline text_variants
# ============================================================================
def _resolve_concept(
    concept_spec,
    rank_override: Optional[int],
    runtime: Runtime,
    slot_pil: Optional[Image.Image] = None,
) -> Tuple[np.ndarray, int, str]:
    """Return (projection_matrix P, effective_rank, source_descriptor).

    `concept_spec` can be:
      - str in CONCEPTS                 → use that built-in
      - str ending in ".npy"            → load that file directly
      - any other str                   → auto-pipeline: LLM-generate variants
                                          (using slot_pil as vision context if
                                          provided), CLIP-encode, cache, then use
      - dict {"text_variants": [...]}   → escape hatch: encode given variants
    """
    if isinstance(concept_spec, str):
        if concept_spec in CONCEPTS:
            fname, default_rank = CONCEPTS[concept_spec]
            npy_path = os.path.join(TEXT_EMBED_DIR, fname)
            text_embeds = np.load(npy_path)
            rank = rank_override if rank_override is not None else default_rank
            source = f"builtin:{concept_spec}"
        elif concept_spec.endswith(".npy") and os.path.exists(concept_spec):
            text_embeds = np.load(concept_spec)
            rank = rank_override if rank_override is not None else 30
            source = f"npy:{concept_spec}"
        else:
            # Free-form concept name: cache lookup or LLM-generate
            text_embeds, default_rank, source = ensure_concept_subspace(
                concept_spec, runtime, slot_pil=slot_pil,
            )
            rank = rank_override if rank_override is not None else default_rank
    elif isinstance(concept_spec, dict) and "text_variants" in concept_spec:
        variants = [str(v).strip() for v in concept_spec["text_variants"] if str(v).strip()]
        if len(variants) < 5:
            raise ValueError(
                f"text_variants needs at least 5 prompts for a usable subspace "
                f"(got {len(variants)}); 30+ recommended."
            )
        text_embeds = runtime.encode_texts(variants)
        rank = rank_override if rank_override is not None else min(30, len(variants))
        source = f"inline:{len(variants)}_variants"
    else:
        raise ValueError(f"Bad concept spec: {concept_spec!r}")

    rank = max(1, min(rank, min(text_embeds.shape)))
    P = compute_projection_matrix(text_embeds, rank)
    return P, rank, source


# ============================================================================
# Flask app
# ============================================================================
OUTPUT_DIR = "outputs/api"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", device=get_runtime().device)


@app.get("/concepts")
def list_concepts():
    """List built-in concepts AND any LLM-cached ones now on disk."""
    builtin_files = {fname for fname, _ in CONCEPTS.values()}
    builtins = [
        {
            "name": name,
            "embeddings_path": os.path.join(TEXT_EMBED_DIR, fname),
            "rank": rank,
            "available": os.path.exists(os.path.join(TEXT_EMBED_DIR, fname)),
            "kind": "builtin",
        }
        for name, (fname, rank) in sorted(CONCEPTS.items())
    ]
    cached = []
    if os.path.isdir(TEXT_EMBED_DIR):
        for fn in sorted(os.listdir(TEXT_EMBED_DIR)):
            if fn.endswith(".npy") and fn not in builtin_files:
                cached.append({
                    "name": fn.removesuffix("_descriptions.npy").removesuffix(".npy"),
                    "embeddings_path": os.path.join(TEXT_EMBED_DIR, fn),
                    "rank": DEFAULT_AUTO_RANK,
                    "available": True,
                    "kind": "auto_generated",
                })
    return jsonify(concepts=builtins + cached)


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.post("/compose")
def compose():
    runtime = get_runtime()

    # ---------------- parse ----------------
    raw_params = request.form.get("params")
    if not raw_params:
        return jsonify(error="missing 'params' form field (JSON string)"), 400
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError as e:
        return jsonify(error=f"invalid JSON in 'params': {e}"), 400

    slots_meta = params.get("slots") or []
    if not slots_meta:
        return jsonify(error="'slots' must be a non-empty list"), 400

    if "base_image" not in request.files:
        return jsonify(error="missing 'base_image' file"), 400

    # ---------------- run (serialized on GPU lock) ----------------
    with runtime.lock:
        # encode base
        base_pil = Image.open(request.files["base_image"].stream)
        base_embed = runtime.encode_image_pil(base_pil)

        # build slots (in-memory equivalent of demo_multi_concept's ConceptSlot list)
        slots: List[Dict] = []
        slot_info: List[Dict] = []
        for i, meta in enumerate(slots_meta):
            img_key = meta.get("image_key", f"slot_{i}")
            if img_key not in request.files:
                return jsonify(error=f"missing file for slot {i} (key '{img_key}')"), 400

            alpha = float(meta.get("alpha", 1.0))
            label = meta.get("name") or f"slot{i}"
            if alpha == 0.0:
                slot_info.append({"name": label, "alpha": 0.0, "skipped": True})
                continue

            # Load the slot image FIRST so we can pass it to the LLM as a vision
            # input when the concept needs to be auto-generated.
            slot_pil = Image.open(request.files[img_key].stream)
            slot_pil.load()  # force decode now; stream is one-shot

            try:
                P, eff_rank, source = _resolve_concept(
                    meta["concept"], meta.get("rank"), runtime, slot_pil=slot_pil,
                )
            except KeyError:
                return jsonify(error=f"slot {i}: missing 'concept' field"), 400
            except ValueError as e:
                return jsonify(error=f"slot {i}: {e}"), 400

            slot_embed = runtime.encode_image_pil(slot_pil)

            # signal_ratio = ‖P(slot_embed)‖ / ‖slot_embed‖
            # Fraction of the slot image's CLIP embedding that lies in the
            # concept subspace. Low (e.g. < 0.10) means the slot image only
            # weakly contains this concept, so even high alpha won't move the
            # output much. UI can flag the slot as "weak signal — try another
            # reference image".
            slot_proj_norm = float(np.linalg.norm(slot_embed @ P))
            slot_full_norm = float(np.linalg.norm(slot_embed))
            signal_ratio = slot_proj_norm / (slot_full_norm + 1e-12)

            slots.append({"embed": slot_embed, "P": P, "alpha": alpha})
            slot_info.append({
                "name": label,
                "alpha": alpha,
                "rank": eff_rank,
                "source": source,
                "signal_ratio": round(signal_ratio, 4),
                "weak_signal": signal_ratio < 0.10,
            })

        if not slots:
            return jsonify(error="no active slots (all alpha=0)"), 400

        # compose (the algorithm)
        final = compose_embedding(base_embed, slots)
        drift = float(
            np.linalg.norm(final - base_embed) / (np.linalg.norm(base_embed) + 1e-12)
        )

        # generate
        prompt = params.get("prompt")
        scale = float(params.get("scale", 1.0))
        seed = int(params.get("seed", 420))
        num_samples = int(params.get("num_samples", 4))
        num_inference_steps = int(params.get("num_inference_steps", 50))

        images = runtime.ip_model.generate(
            clip_image_embeds=torch.from_numpy(final),
            prompt=prompt,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=7.5,
            scale=scale,
        )

    # ---------------- save + respond ----------------
    run_id = uuid.uuid4().hex[:8]
    ts = int(time.time())
    out_files = []
    for i, im in enumerate(images):
        fn = f"{ts}_{run_id}_{i + 1}.png"
        im.save(os.path.join(OUTPUT_DIR, fn))
        out_files.append(fn)

    return jsonify({
        "drift": round(drift, 4),
        "drift_warn": drift > 0.6,   # output may go off-distribution
        "slots": slot_info,
        "num_samples": len(images),
        "files": out_files,
        "urls": [f"/outputs/{fn}" for fn in out_files],
    })


if __name__ == "__main__":
    get_runtime()  # eager-load so startup failures surface immediately
    app.run(host="0.0.0.0", port=12100, threaded=True)
