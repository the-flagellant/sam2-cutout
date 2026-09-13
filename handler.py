"""
RunPod Serverless worker — SAM 2.1 cutouts (v2).

Back-compat: image_url | image_b64, mode center|auto|point|box,
returns cutout_png_b64, mask_png_b64, score.

New:
  - box + points + negative points in one call (mode=combined or just pass both)
  - keep_canvas / crop=false + source bbox
  - soft alpha (feather) + hole-fill + keep-largest-CC + morph close
  - EXIF transpose, optional rotate
  - box_normalized 0–1
  - max_side downsample → upsample mask
  - return_multimasks
  - batch via images: [...]
  - prefer_standing for auto
  - structured errors
"""
from __future__ import annotations

import base64
import io
import os
import traceback
from typing import Any

import cv2
import numpy as np
import requests
import runpod
import torch
from PIL import Image, ImageOps

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = os.getenv("SAM2_SIZE", "large")
CKPT_DIR = os.getenv("SAM2_CKPT_DIR", "/models")
VERSION = "2.0.0"

SIZE_MAP = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}

_cfg, _ckpt_name = SIZE_MAP.get(MODEL_SIZE, SIZE_MAP["large"])
_ckpt = os.path.join(CKPT_DIR, _ckpt_name)

print(f"[sam2] v{VERSION} loading {MODEL_SIZE} on {DEVICE} from {_ckpt}", flush=True)
_sam = build_sam2(_cfg, _ckpt, device=DEVICE)
_predictor = SAM2ImagePredictor(_sam)
_auto = SAM2AutomaticMaskGenerator(
    _sam,
    points_per_side=int(os.getenv("SAM2_POINTS_PER_SIDE", "16")),
    pred_iou_thresh=0.82,
    stability_score_thresh=0.88,
    min_mask_region_area=400,
)
print("[sam2] ready", flush=True)


class JobError(Exception):
    def __init__(self, code: str, message: str, http: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http = http


def _png_b64(img: Image.Image, compress: int = 6) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fetch_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; sam2-runpod/2.0; +https://runpod.io) "
            "AppleWebKit/537.36 Chrome/120.0.0.0"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://x.com/",
    }
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45, headers=headers, allow_redirects=True)
            r.raise_for_status()
            if len(r.content) < 64:
                raise JobError("empty_image", "Fetched image was empty")
            return r.content
        except JobError:
            raise
        except Exception as e:
            last = e
    raise JobError("fetch_failed", f"Could not fetch image_url: {last}", 502)


def _load_pil(inp: dict) -> Image.Image:
    if inp.get("image_b64"):
        try:
            raw = base64.b64decode(inp["image_b64"])
        except Exception as e:
            raise JobError("bad_b64", f"image_b64 is not valid base64: {e}")
        im = Image.open(io.BytesIO(raw))
    elif inp.get("image_url"):
        im = Image.open(io.BytesIO(_fetch_bytes(str(inp["image_url"]))))
    else:
        raise JobError("no_image", "Provide image_url or image_b64")
    im = ImageOps.exif_transpose(im)
    rot = inp.get("rotate", 0)
    if rot in (90, 180, 270, "90", "180", "270"):
        im = im.rotate(int(rot), expand=True)
    return im.convert("RGB")


def _maybe_downscale(rgb: Image.Image, max_side: int | None) -> tuple[Image.Image, float]:
    if not max_side:
        return rgb, 1.0
    w, h = rgb.size
    m = max(w, h)
    if m <= max_side:
        return rgb, 1.0
    scale = max_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return rgb.resize((nw, nh), Image.Resampling.BILINEAR), scale


def _upsample_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _norm_box(box, w: int, h: int, normalized: bool) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise JobError("bad_box", "box must be [x1,y1,x2,y2]")
    if normalized or float(np.max(np.abs(b))) <= 1.5:
        b = np.array([b[0] * w, b[1] * h, b[2] * w, b[3] * h], dtype=np.float32)
    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 1, w))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 1, h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _norm_points(pts, w: int, h: int, normalized: bool) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 2:
        raise JobError("bad_points", "points must be [[x,y], ...]")
    if normalized or float(np.max(np.abs(a))) <= 1.5:
        a = a * np.array([w, h], dtype=np.float32)
    a[:, 0] = np.clip(a[:, 0], 0, w - 1)
    a[:, 1] = np.clip(a[:, 1], 0, h - 1)
    return a


def _keep_largest_cc(m: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    # skip background 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.where(labels == keep, m, 0)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    bin8 = (m > 0).astype(np.uint8)
    h, w = bin8.shape
    flood = bin8.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (1 - flood).astype(np.uint8)
    return np.where((bin8 + holes) > 0, np.maximum(m, 1.0), m)


def _refine_alpha(
    mask: np.ndarray,
    *,
    hole_fill: bool = True,
    keep_largest: bool = True,
    close: int = 3,
    erode: int = 0,
    feather: int = 4,
) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    bin8 = (m > 0.5).astype(np.uint8)
    if close and close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close * 2 + 1, close * 2 + 1))
        bin8 = cv2.morphologyEx(bin8, cv2.MORPH_CLOSE, k)
    if keep_largest:
        bin8 = _keep_largest_cc(bin8).astype(np.uint8)
        bin8 = (bin8 > 0).astype(np.uint8)
    if hole_fill:
        bin8 = (_fill_holes(bin8) > 0).astype(np.uint8)
    if erode and erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        bin8 = cv2.erode(bin8, k)
    if feather and feather > 0:
        # inner distance fade so the edge is soft without growing the subject
        dist = cv2.distanceTransform(bin8, cv2.DIST_L2, 3)
        alpha = np.clip(dist / float(feather), 0.0, 1.0)
        # preserve interior 1s; blur helps hair
        blur = cv2.GaussianBlur(bin8.astype(np.float32), (0, 0), max(0.6, feather / 3.0))
        alpha = np.maximum(alpha, np.clip(blur, 0, 1) * bin8)
        alpha = np.clip(alpha, 0, 1)
    else:
        alpha = bin8.astype(np.float32)
    return alpha


def _decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pull fringe toward nearby foreground color so halos die."""
    a = alpha[..., None]
    fg = rgb.astype(np.float32)
    # background estimate: heavily blurred original
    bg = cv2.GaussianBlur(fg, (0, 0), 6)
    fringe = (alpha > 0.02) & (alpha < 0.92)
    if not np.any(fringe):
        return rgb
    out = fg.copy()
    t = ((1.0 - alpha) * 0.65)[..., None]
    out[fringe] = np.clip(fg[fringe] - t[fringe] * (bg[fringe] - 16), 0, 255)
    return out.astype(np.uint8)


def _bbox_from_alpha(alpha: np.ndarray, pad_pct: float = 0.0) -> list[int]:
    ys, xs = np.where(alpha > 0.05)
    if len(xs) == 0:
        return [0, 0, alpha.shape[1], alpha.shape[0]]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if pad_pct:
        pw = int(round((x2 - x1) * pad_pct))
        ph = int(round((y2 - y1) * pad_pct))
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(alpha.shape[1], x2 + pw)
        y2 = min(alpha.shape[0], y2 + ph)
    return [x1, y1, x2, y2]


def _compose(rgb: Image.Image, alpha: np.ndarray, crop: bool, pad_pct: float) -> tuple[Image.Image, list[int]]:
    arr = np.asarray(rgb).copy()
    arr = _decontaminate(arr, alpha)
    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, a8])
    bbox = _bbox_from_alpha(alpha, pad_pct=pad_pct)
    im = Image.fromarray(rgba, "RGBA")
    if crop:
        im = im.crop(tuple(bbox))
    return im, bbox


def _auto_masks(rgb_small: Image.Image, prefer_standing: bool) -> list[dict]:
    arr = np.asarray(rgb_small)
    h, w = arr.shape[:2]
    raw = _auto.generate(arr)
    if not raw:
        raise JobError("no_mask", "SAM2 auto produced no masks")
    area = h * w
    out = []
    cx, cy = w / 2.0, h / 2.0
    for m in raw:
        seg = m["segmentation"].astype(np.float32)
        a = int(m["area"])
        if a < 0.015 * area or a > 0.94 * area:
            continue
        ys, xs = np.where(seg > 0.5)
        if len(xs) == 0:
            continue
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        aspect = bh / max(1.0, float(bw))
        mx, my = float(xs.mean()), float(ys.mean())
        dist = ((mx - cx) / w) ** 2 + ((my - cy) / h) ** 2
        score = float(m["predicted_iou"]) * (1.0 + 0.35 * max(0.0, aspect - 1.0) if prefer_standing else 1.0)
        score *= max(0.25, 1.0 - dist)
        out.append({"mask": seg, "score": score, "area": a, "iou": float(m["predicted_iou"]), "aspect": aspect})
    if not out:
        best = max(raw, key=lambda m: float(m["predicted_iou"]) * int(m["area"]))
        out = [{"mask": best["segmentation"].astype(np.float32), "score": float(best["predicted_iou"]),
                "area": int(best["area"]), "iou": float(best["predicted_iou"]), "aspect": 1.0}]
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def _prompted_masks(rgb_small: Image.Image, inp: dict) -> list[dict]:
    arr = np.asarray(rgb_small)
    h, w = arr.shape[:2]
    _predictor.set_image(arr)
    kwargs: dict[str, Any] = {"multimask_output": True}
    box = inp.get("box")
    pts = inp.get("points")
    labs = inp.get("point_labels")
    normalized = bool(inp.get("box_normalized") or inp.get("normalized"))
    mode = str(inp.get("mode", "center")).lower()

    if box:
        kwargs["box"] = _norm_box(box, w, h, normalized)
    if pts:
        kwargs["point_coords"] = _norm_points(pts, w, h, normalized)
        if labs is None:
            labs = [1] * len(pts)
        kwargs["point_labels"] = np.asarray(labs, dtype=np.int32)
        if len(kwargs["point_labels"]) != len(pts):
            raise JobError("bad_points", "point_labels length must match points")
    if not box and not pts:
        if mode == "box":
            raise JobError("bad_box", "box mode needs box=[x1,y1,x2,y2]")
        if mode == "point":
            raise JobError("bad_points", "point mode needs points=[[x,y],...]")
        # center fallback
        kwargs["point_coords"] = np.array([[w / 2.0, h / 2.0]], dtype=np.float32)
        kwargs["point_labels"] = np.array([1], dtype=np.int32)

    masks, scores, _logits = _predictor.predict(**kwargs)
    out = []
    for i, (msk, sc) in enumerate(zip(masks, scores)):
        m = msk.astype(np.float32)
        out.append({"mask": m, "score": float(sc), "area": int((m > 0.5).sum()), "iou": float(sc), "index": i})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def _pick_standing(cands: list[dict]) -> dict:
    """Prefer a tall full-body mask over a high-score torso."""
    if not cands:
        raise JobError("no_mask", "No mask candidates")
    ranked = []
    for c in cands:
        m = c["mask"]
        ys, xs = np.where(m > 0.5)
        if len(xs) == 0:
            continue
        aspect = (ys.max() - ys.min() + 1) / max(1.0, float(xs.max() - xs.min() + 1))
        cover = float(c.get("area") or (m > 0.5).sum())
        ranked.append((aspect * 0.55 + (cover ** 0.5) * 1e-3 + c["score"] * 0.45, c))
    if not ranked:
        return cands[0]
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[0][1]


def process_one(inp: dict) -> dict:
    mode = str(inp.get("mode", "center")).lower()
    if mode in ("hybrid",):
        mode = "combined"
    if mode not in ("auto", "center", "point", "box", "combined"):
        raise JobError("bad_mode", f"unknown mode {mode}")

    rgb = _load_pil(inp)
    src_w, src_h = rgb.size
    max_side = inp.get("max_side", 1600)
    try:
        max_side = int(max_side) if max_side else None
    except (TypeError, ValueError):
        max_side = 1600
    small, scale = _maybe_downscale(rgb, max_side)

    prefer_standing = bool(inp.get("prefer_standing", True))
    if mode == "auto":
        cands = _auto_masks(small, prefer_standing=prefer_standing)
    else:
        cands = _prompted_masks(small, inp)

    want_multi = bool(inp.get("return_multimasks") or inp.get("multimasks"))
    multi_n = int(inp.get("multimask_max", 3))
    chosen = _pick_standing(cands) if prefer_standing else cands[0]
    mask_small = chosen["mask"]
    mask = _upsample_mask(mask_small, (src_w, src_h))

    feather = int(inp.get("feather", 4))
    close = int(inp.get("close", 3))
    erode = int(inp.get("erode", 0))
    hole_fill = bool(inp.get("hole_fill", True))
    keep_largest = bool(inp.get("keep_largest", True))
    pad_pct = float(inp.get("pad_pct", 0.0))
    crop = inp.get("crop")
    if crop is None:
        crop = not bool(inp.get("keep_canvas", False))
    crop = bool(crop)

    alpha = _refine_alpha(
        mask,
        hole_fill=hole_fill,
        keep_largest=keep_largest,
        close=close,
        erode=erode,
        feather=feather,
    )
    if float(alpha.max()) < 0.05:
        raise JobError("empty_mask", "Mask was empty after refine")

    cut, bbox = _compose(rgb, alpha, crop=crop, pad_pct=pad_pct)
    mask_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), "L")

    out = {
        "ok": True,
        "version": VERSION,
        "mode": mode,
        "model": f"sam2.1_hiera_{MODEL_SIZE}",
        "device": DEVICE,
        "image_size": [src_w, src_h],
        "cutout_size": list(cut.size),
        "score": round(float(chosen["score"]), 4),
        "bbox": bbox,
        "bbox_norm": [round(bbox[0] / src_w, 5), round(bbox[1] / src_h, 5),
                      round(bbox[2] / src_w, 5), round(bbox[3] / src_h, 5)],
        "canvas": "cropped" if crop else "full",
        "cutout_png_b64": _png_b64(cut),
        "mask_png_b64": _png_b64(mask_img),
    }
    if want_multi:
        extras = []
        for c in cands[:multi_n]:
            mm = _upsample_mask(c["mask"], (src_w, src_h))
            aa = _refine_alpha(mm, hole_fill=hole_fill, keep_largest=keep_largest,
                               close=close, erode=erode, feather=max(1, feather // 2))
            extras.append({
                "score": round(float(c["score"]), 4),
                "area": int((aa > 0.5).sum()),
                "bbox": _bbox_from_alpha(aa),
                "mask_png_b64": _png_b64(Image.fromarray(np.clip(aa * 255, 0, 255).astype(np.uint8), "L")),
            })
        out["masks"] = extras
    return out


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    try:
        if inp.get("images"):
            results = []
            shared = {k: v for k, v in inp.items() if k not in ("images", "defaults")}
            if isinstance(inp.get("defaults"), dict):
                shared.update(inp["defaults"])
            for i, item in enumerate(inp["images"]):
                merged = dict(shared)
                merged.update(item or {})
                try:
                    results.append(process_one(merged))
                except JobError as e:
                    results.append({"ok": False, "error_code": e.code, "error": e.message, "index": i})
                except Exception as e:
                    results.append({"ok": False, "error_code": "internal", "error": str(e), "index": i})
            ok_n = sum(1 for r in results if r.get("ok"))
            return {"ok": ok_n == len(results), "version": VERSION, "count": len(results), "ok_count": ok_n, "results": results}
        return process_one(inp)
    except JobError as e:
        return {"ok": False, "error_code": e.code, "error": e.message}
    except Exception as e:
        return {"ok": False, "error_code": "internal", "error": str(e), "trace": traceback.format_exc()[-1500:]}


runpod.serverless.start({"handler": handler})
