"""
Multi-image, multi-concept fusion demo for IP-Composer.

Algorithm (alpha-aware extension of the original paper):

    final = base + Σᵢ αᵢ · ( Pᵢ(concept_i) - Pᵢ(base) )

    where:
      - base, concept_i  : CLIP image embeddings, shape (1, D)
      - Pᵢ               : projection matrix for concept i (D, D),
                           built from SVD of a set of text embeddings
      - αᵢ               : per-concept weight
                             αᵢ = 1   ⇒ full replace within subspace (paper default)
                             0<α<1    ⇒ partial fuse
                             αᵢ = 0   ⇒ slot disabled
                             αᵢ < 0   ⇒ push away from concept_i (suppression)

The user manually specifies the base image. Each concept slot independently
declares its source image, the subspace (either a precomputed .npy of text
embeddings, or a CSV of text variants encoded on-the-fly), and a weight.

Two run modes:
    --test                Algorithmic correctness checks (CPU-only, no IP-Adapter).
    --config <yaml/json>  Full generation: encode images, build projections,
                          compose, generate via SDXL + IP-Adapter.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np


# ============================================================================
# Config
# ============================================================================
@dataclass
class ConceptSlot:
    image_path: str
    subspace_source: str       # ".npy" (precomputed text embeddings) or ".csv" (text variants)
    rank: int = 30
    alpha: float = 1.0
    name: str = ""             # human-readable label, used for logging only


@dataclass
class DemoConfig:
    base_image_path: str
    concept_slots: List[ConceptSlot]
    output_path: str = "outputs/multi_concept.png"
    prompt: Optional[str] = None
    scale: float = 1.0
    seed: int = 420
    num_inference_steps: int = 50
    num_samples: int = 1


# ============================================================================
# Core math (pure numpy, no CLIP / no IP-Adapter — safe to unit-test)
# ============================================================================
def compute_projection_matrix(text_embeds: np.ndarray, rank: int) -> np.ndarray:
    """SVD-based projection onto the top-`rank` right-singular subspace.

    text_embeds: (N, D) — rows are CLIP text embeddings of concept variants.
    Returns P: (D, D) such that x @ P projects x onto span(top-rank V).
    P is symmetric and idempotent (P @ P == P).
    """
    if text_embeds.ndim != 2:
        raise ValueError(f"text_embeds must be 2-D, got shape {text_embeds.shape}")
    if rank < 1 or rank > min(text_embeds.shape):
        raise ValueError(f"rank {rank} out of range for shape {text_embeds.shape}")
    _, _, v = np.linalg.svd(text_embeds, full_matrices=False)
    v_r = v[:rank]                # (rank, D)
    return v_r.T @ v_r            # (D, D)


def project(embed: np.ndarray, P: np.ndarray) -> np.ndarray:
    return embed @ P


def compose_embedding(
    base_embed: np.ndarray,
    slots: List[Dict],
) -> np.ndarray:
    """Apply the alpha-aware composition.

    Each slot is a dict with keys:
        embed : (1, D) numpy array — concept image embedding
        P     : (D, D) projection matrix
        alpha : float — weight (positive = fuse, negative = suppress)
    """
    out = base_embed.copy()
    for s in slots:
        delta = project(s["embed"], s["P"]) - project(base_embed, s["P"])
        out = out + float(s["alpha"]) * delta
    return out


# ============================================================================
# Subspace loading
# ============================================================================
def load_subspace_text_embeds(
    source: str,
    encode_texts: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> np.ndarray:
    """Load text embeddings used to build a concept's projection subspace.

    - .npy : returns the array directly.
    - .csv : reads first column as text variants, encodes via `encode_texts`.
    """
    if source.endswith(".npy"):
        return np.load(source)
    if source.endswith(".csv"):
        if encode_texts is None:
            raise RuntimeError(
                "CSV subspace_source requires a CLIP text encoder. "
                "This is unavailable in --test mode; use a .npy source instead."
            )
        rows: List[str] = []
        with open(source, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for r in reader:
                if r and r[0].strip():
                    rows.append(r[0].strip())
        if len(rows) < 30:
            print(f"[warn] only {len(rows)} text variants in {source}; "
                  f"recommend ≥100 for stable subspaces")
        return encode_texts(rows)
    raise ValueError(f"Unsupported subspace_source extension: {source}")


# ============================================================================
# Generation pipeline (lazy imports — heavy deps only loaded on demand)
# ============================================================================
def _load_runtime():
    import torch
    import open_clip
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download
    from IP_Adapter import IPAdapterXL
    return torch, open_clip, Image, StableDiffusionXLPipeline, hf_hub_download, IPAdapterXL


def run_generation(cfg: DemoConfig) -> None:
    torch, open_clip, Image, SDXLPipeline, hf_hub_download, IPAdapterXL = _load_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- CLIP ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    clip_model.to(device)
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def encode_image(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        x = preprocess(img)[None].to(device)
        with torch.no_grad():
            e = clip_model.encode_image(x)
        return e.cpu().numpy()

    def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = tokenizer(texts[i:i + batch_size]).to(device)
            with torch.no_grad():
                e = clip_model.encode_text(tok)
            out.append(e.cpu().numpy())
        return np.vstack(out)

    # --- IP-Adapter / SDXL ---
    pipe = SDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    ip_ckpt = hf_hub_download(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        filename="ip-adapter_sdxl_vit-h.bin",
    )
    ip_model = IPAdapterXL(pipe, "h94/IP-Adapter", "models/image_encoder", ip_ckpt, device)

    # --- Encode base + concept images ---
    print(f"[base] {cfg.base_image_path}")
    base_embed = encode_image(cfg.base_image_path)

    slots: List[Dict] = []
    for i, slot_cfg in enumerate(cfg.concept_slots):
        label = slot_cfg.name or f"slot{i}"
        if slot_cfg.alpha == 0.0:
            print(f"[skip] {label} (alpha=0)")
            continue
        print(f"[slot] {label}: image={slot_cfg.image_path}, "
              f"subspace={slot_cfg.subspace_source}, rank={slot_cfg.rank}, alpha={slot_cfg.alpha}")
        text_embeds = load_subspace_text_embeds(slot_cfg.subspace_source, encode_texts)
        P = compute_projection_matrix(text_embeds, slot_cfg.rank)
        concept_embed = encode_image(slot_cfg.image_path)
        slots.append({"embed": concept_embed, "P": P, "alpha": slot_cfg.alpha, "name": label})

    if not slots:
        raise RuntimeError("No active concept slots (all alpha=0).")

    # --- Compose ---
    final = compose_embedding(base_embed, slots)

    # --- Sanity log: how far did we drift? ---
    drift = float(np.linalg.norm(final - base_embed) / (np.linalg.norm(base_embed) + 1e-12))
    print(f"[compose] relative drift |final - base| / |base| = {drift:.3f}")
    if drift > 0.6:
        print("[warn] large drift; output may go off-distribution. "
              "Consider lowering alphas or rank.")

    # --- Generate ---
    clip_embeds = torch.from_numpy(final)
    images = ip_model.generate(
        clip_image_embeds=clip_embeds,
        prompt=cfg.prompt,
        num_samples=cfg.num_samples,
        num_inference_steps=cfg.num_inference_steps,
        seed=cfg.seed,
        guidance_scale=7.5,
        scale=cfg.scale,
    )

    out_dir = os.path.dirname(cfg.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        path = cfg.output_path
        if cfg.num_samples > 1:
            stem, ext = os.path.splitext(path)
            path = f"{stem}_{i + 1}{ext}"
        im.save(path)
        print(f"[saved] {path}")


# ============================================================================
# Tests — pure numpy, no GPU, no model downloads
# ============================================================================
def _rand_text_embeds(n: int = 200, d: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


def test_projection_idempotent():
    """An orthogonal projector satisfies P @ P == P."""
    P = compute_projection_matrix(_rand_text_embeds(seed=0), rank=30)
    err = float(np.max(np.abs(P @ P - P)))
    print(f"  max|PP - P| = {err:.2e}")
    assert err < 1e-8, "P is not idempotent"


def test_projection_symmetric():
    P = compute_projection_matrix(_rand_text_embeds(seed=1), rank=30)
    err = float(np.max(np.abs(P - P.T)))
    print(f"  max|P - P.T| = {err:.2e}")
    assert err < 1e-10, "P is not symmetric"


def test_projection_nonexpansive():
    """||x P|| ≤ ||x|| for any x."""
    P = compute_projection_matrix(_rand_text_embeds(seed=2), rank=30)
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.standard_normal(1024)
        assert np.linalg.norm(x @ P) <= np.linalg.norm(x) + 1e-10


def test_compose_zero_alpha_is_identity():
    """With every alpha=0, output must equal base exactly."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=3), rank=30)
    slots = [
        {"embed": rng.standard_normal((1, 1024)), "P": P, "alpha": 0.0},
        {"embed": rng.standard_normal((1, 1024)), "P": P, "alpha": 0.0},
    ]
    out = compose_embedding(base, slots)
    err = float(np.max(np.abs(out - base)))
    print(f"  max|out - base| = {err:.2e}")
    assert err < 1e-10


def test_compose_self_concept_is_identity():
    """If concept_embed == base, the slot has no effect regardless of alpha."""
    rng = np.random.default_rng(4)
    base = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=4), rank=30)
    slots = [{"embed": base.copy(), "P": P, "alpha": 1.0}]
    out = compose_embedding(base, slots)
    err = float(np.max(np.abs(out - base)))
    print(f"  max|out - base| = {err:.2e}")
    assert err < 1e-10


def test_compose_alpha1_replaces_subspace():
    """alpha=1 ⇒ P(out) == P(concept) (full replace inside subspace)."""
    rng = np.random.default_rng(5)
    base = rng.standard_normal((1, 1024))
    concept = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=5), rank=30)
    slots = [{"embed": concept, "P": P, "alpha": 1.0}]
    out = compose_embedding(base, slots)
    err = float(np.max(np.abs(out @ P - concept @ P)))
    print(f"  max|P(out) - P(concept)| = {err:.2e}")
    assert err < 1e-8


def test_compose_complement_unchanged():
    """The orthogonal complement (I - P) of the subspace stays = base's part."""
    rng = np.random.default_rng(6)
    base = rng.standard_normal((1, 1024))
    concept = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=6), rank=30)
    I = np.eye(1024)
    slots = [{"embed": concept, "P": P, "alpha": 1.0}]
    out = compose_embedding(base, slots)
    err = float(np.max(np.abs(out @ (I - P) - base @ (I - P))))
    print(f"  max|complement diff| = {err:.2e}")
    assert err < 1e-8


def test_compose_negative_alpha_is_pushaway():
    """alpha=-1 ⇒ P(out) - P(base) == -(P(concept) - P(base))."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((1, 1024))
    concept = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=7), rank=30)
    slots = [{"embed": concept, "P": P, "alpha": -1.0}]
    out = compose_embedding(base, slots)
    expected = 2.0 * (base @ P) - (concept @ P)   # = base@P - (concept@P - base@P)
    err = float(np.max(np.abs(out @ P - expected)))
    print(f"  max|deviation| = {err:.2e}")
    assert err < 1e-8


def test_two_concepts_from_same_image_are_additive():
    """Two slots sharing the same source image with non-overlapping subspaces
    behave the same whether stacked in one call or applied sequentially."""
    rng = np.random.default_rng(8)
    base = rng.standard_normal((1, 1024))
    src = rng.standard_normal((1, 1024))
    # Build two genuinely orthogonal subspaces by partitioning a random orthonormal basis.
    Q, _ = np.linalg.qr(rng.standard_normal((1024, 1024)))
    P1 = Q[:, :30] @ Q[:, :30].T
    P2 = Q[:, 30:60] @ Q[:, 30:60].T
    # Joint
    out_joint = compose_embedding(
        base,
        [{"embed": src, "P": P1, "alpha": 1.0},
         {"embed": src, "P": P2, "alpha": 1.0}],
    )
    # Sequential
    step1 = compose_embedding(base, [{"embed": src, "P": P1, "alpha": 1.0}])
    out_seq = compose_embedding(step1, [{"embed": src, "P": P2, "alpha": 1.0}])
    err = float(np.max(np.abs(out_joint - out_seq)))
    print(f"  joint vs sequential (orthogonal subspaces): max diff = {err:.2e}")
    assert err < 1e-8, "orthogonal subspaces should commute"


def test_real_text_embeddings_subspace_selective():
    """Concept-text rows should project into their own subspace much more than random vectors do."""
    npy = "text_embeddings/age_descriptions.npy"
    if not os.path.exists(npy):
        print(f"  [skipped] {npy} not present")
        return
    text_embeds = np.load(npy).astype(np.float64)
    P = compute_projection_matrix(text_embeds, rank=30)
    in_ratio = (np.linalg.norm(text_embeds @ P, axis=1)
                / (np.linalg.norm(text_embeds, axis=1) + 1e-12)).mean()
    rng = np.random.default_rng(9)
    rand = rng.standard_normal((200, text_embeds.shape[1])).astype(np.float64)
    rand_ratio = (np.linalg.norm(rand @ P, axis=1)
                  / (np.linalg.norm(rand, axis=1) + 1e-12)).mean()
    print(f"  concept text ratio={in_ratio:.3f}, random ratio={rand_ratio:.3f}")
    assert in_ratio > 2 * rand_ratio, "concept subspace not selectively capturing concept texts"


def test_drift_grows_with_alpha():
    """As |alpha| grows, output drifts further from base monotonically."""
    rng = np.random.default_rng(10)
    base = rng.standard_normal((1, 1024))
    concept = rng.standard_normal((1, 1024))
    P = compute_projection_matrix(_rand_text_embeds(seed=10), rank=30)
    drifts = []
    for a in [0.0, 0.25, 0.5, 1.0, 1.5]:
        out = compose_embedding(base, [{"embed": concept, "P": P, "alpha": a}])
        drifts.append(np.linalg.norm(out - base))
    print(f"  drift sequence: {[f'{d:.3f}' for d in drifts]}")
    for i in range(1, len(drifts)):
        assert drifts[i] >= drifts[i - 1] - 1e-10, "drift should be non-decreasing in |alpha|"


_TESTS = [
    test_projection_idempotent,
    test_projection_symmetric,
    test_projection_nonexpansive,
    test_compose_zero_alpha_is_identity,
    test_compose_self_concept_is_identity,
    test_compose_alpha1_replaces_subspace,
    test_compose_complement_unchanged,
    test_compose_negative_alpha_is_pushaway,
    test_two_concepts_from_same_image_are_additive,
    test_real_text_embeddings_subspace_selective,
    test_drift_grows_with_alpha,
]


def run_tests() -> bool:
    passed = 0
    for t in _TESTS:
        print(f"[{t.__name__}]")
        try:
            t()
            print("  OK")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(_TESTS)} passed")
    return passed == len(_TESTS)


# ============================================================================
# CLI
# ============================================================================
def _load_config_dict(path: str) -> dict:
    if path.endswith((".yml", ".yaml")):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    if path.endswith(".json"):
        import json
        with open(path) as f:
            return json.load(f)
    raise ValueError(f"Config must be .yaml or .json, got {path}")


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config")
    parser.add_argument("--test", action="store_true", help="Run algorithm tests and exit")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    if not args.config:
        parser.error("either --test or --config is required")

    raw = _load_config_dict(args.config)
    slots_raw = raw.pop("concept_slots", [])
    slots = [ConceptSlot(**s) for s in slots_raw]
    return DemoConfig(concept_slots=slots, **raw)


if __name__ == "__main__":
    cfg = parse_args()
    run_generation(cfg)
