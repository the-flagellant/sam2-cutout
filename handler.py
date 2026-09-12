"""
RunPod Serverless worker — SAM 2.1 image cutouts.

Load once outside the handler. Input image URL or base64.
Modes:
  auto    — SAM2AutomaticMaskGenerator, keep largest non-full-frame mask
  center  — one positive click at image center (best for portraits)
  point   — explicit points + labels
  box     — one [x1, y1, x2, y2] box in pixel coords
Returns RGBA PNG (base64) + mask PNG (base64) + score/area.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import requests
import runpod
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = os.getenv("SAM2_SIZE", "large")  # tiny | small | base_plus | large
CKPT_DIR = os.getenv("SAM2_CKPT_DIR", "/models")

SIZE_MAP = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}

_cfg, _ckpt_name = SIZE_MAP.get(MODEL_SIZE, SIZE_MAP["large"])
_ckpt = os.path.join(CKPT_DIR, _ckpt_name)

print(f"[sam2] loading {MODEL_SIZE} on {DEVICE} from {_ckpt}", flush=True)
_sam = build_sam2(_cfg, _ckpt, device=DEVICE)
_predictor = SAM2ImagePredictor(_sam)
_auto = SAM2AutomaticMaskGenerator(
    _sam,
    points_per_side=16,
    pred_iou_thresh=0.82,
    stability_score_thresh=0.88,
    min_mask_region_area=400,
)
print("[sam2] ready", flush=True)


def _load_image(inp: dict) -> Image.Image:
    if inp.get("image_b64"):
        raw = base64.b64decode(inp["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    url = inp.get("image_url")
    if not url:
        raise ValueError("Provide image_url or image_b64")
    r = requests.get(url, timeout=30, headers={"User-Agent": "sam2-runpod/1.0"})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _apply_mask(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    mask_u8 = (mask.astype(np.float32) > 0.5).astype(np.uint8) * 255
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_u8, mode="L"))
    bb = rgba.getchannel("A").getbbox()
    if bb:
        rgba = rgba.crop(bb)
    return rgba


def _best_auto_mask(rgb: Image.Image) -> tuple[np.ndarray, float]:
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    masks = _auto.generate(arr)
    if not masks:
        raise RuntimeError("SAM2 auto produced no masks")
    area = h * w
    ranked = []
    for m in masks:
        a = int(m["area"])
        if a < 0.02 * area or a > 0.92 * area:
            continue
        ranked.append((float(m["predicted_iou"]) * a, m))
    if not ranked:
        ranked = [(float(m["predicted_iou"]) * int(m["area"]), m) for m in masks]
    ranked.sort(key=lambda t: t[0], reverse=True)
    best = ranked[0][1]
    return best["segmentation"].astype(np.uint8), float(best["predicted_iou"])


def _predict_prompted(rgb: Image.Image, inp: dict, mode: str) -> tuple[np.ndarray, float]:
    arr = np.asarray(rgb)
    _predictor.set_image(arr)
    kwargs: dict[str, Any] = {"multimask_output": True}
    if mode == "center":
        h, w = arr.shape[:2]
        kwargs["point_coords"] = np.array([[w / 2.0, h / 2.0]], dtype=np.float32)
        kwargs["point_labels"] = np.array([1], dtype=np.int32)
    elif mode == "point":
        pts = inp.get("points")
        labs = inp.get("point_labels")
        if not pts or not labs:
            raise ValueError("point mode needs points=[[x,y],...] and point_labels=[1 or 0,...]")
        kwargs["point_coords"] = np.asarray(pts, dtype=np.float32)
        kwargs["point_labels"] = np.asarray(labs, dtype=np.int32)
    elif mode == "box":
        box = inp.get("box")
        if not box or len(box) != 4:
            raise ValueError("box mode needs box=[x1,y1,x2,y2] in pixels")
        kwargs["box"] = np.asarray(box, dtype=np.float32)
    else:
        raise ValueError(f"unknown mode {mode}")
    masks, scores, _ = _predictor.predict(**kwargs)
    idx = int(np.argmax(scores))
    return masks[idx].astype(np.uint8), float(scores[idx])


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    mode = str(inp.get("mode", "center")).lower()
    rgb = _load_image(inp)
    if mode == "auto":
        mask, score = _best_auto_mask(rgb)
    else:
        mask, score = _predict_prompted(rgb, inp, mode)
    cut = _apply_mask(rgb, mask)
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    return {
        "ok": True,
        "mode": mode,
        "model": f"sam2.1_hiera_{MODEL_SIZE}",
        "device": DEVICE,
        "image_size": list(rgb.size),
        "cutout_size": list(cut.size),
        "score": round(score, 4),
        "cutout_png_b64": _png_b64(cut),
        "mask_png_b64": _png_b64(mask_img),
    }


runpod.serverless.start({"handler": handler})
