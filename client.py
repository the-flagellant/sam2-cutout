#!/usr/bin/env python3
"""Call the SAM 2.1 RunPod endpoint (v2) and save cutout PNG(s)."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import requests


def _decode_png(b64: str, path: str) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def main() -> int:
    p = argparse.ArgumentParser(description="SAM 2.1 RunPod client v2")
    p.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    p.add_argument("--image-url")
    p.add_argument("--image", help="local image path (sent as base64)")
    p.add_argument("--images", nargs="*", help="batch of local paths")
    p.add_argument("--mode", default="center",
                   choices=["center", "auto", "point", "box", "combined"])
    p.add_argument("--box", help="x1,y1,x2,y2 (pixels or 0-1 if --normalized)")
    p.add_argument("--points", help="x,y;x,y;...  (add --neg for those indices via labels)")
    p.add_argument("--point-labels", help="1,0,1,... matching --points")
    p.add_argument("--normalized", action="store_true", help="boxes/points are 0-1")
    p.add_argument("--keep-canvas", action="store_true", help="do not tight-crop")
    p.add_argument("--feather", type=int, default=4)
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--multimasks", action="store_true")
    p.add_argument("--async", dest="async_run", action="store_true",
                   help="use /run + poll instead of /runsync")
    p.add_argument("--out", default="cutout.png")
    args = p.parse_args()
    if not args.endpoint_id or not args.api_key:
        print("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
        return 2

    inp: dict = {}
    if args.images:
        items = []
        for path in args.images:
            with open(path, "rb") as f:
                items.append({"image_b64": base64.b64encode(f.read()).decode("ascii")})
        inp = {"mode": args.mode, "images": items}
    else:
        inp = {"mode": args.mode}
        if args.image_url:
            inp["image_url"] = args.image_url
        elif args.image:
            with open(args.image, "rb") as f:
                inp["image_b64"] = base64.b64encode(f.read()).decode("ascii")
        else:
            print("Need --image-url, --image, or --images", file=sys.stderr)
            return 2

    if args.box:
        inp["box"] = [float(x) for x in args.box.split(",")]
        if args.mode == "center":
            inp["mode"] = "combined"
    if args.points:
        pts = []
        for pair in args.points.split(";"):
            x, y = pair.split(",")
            pts.append([float(x), float(y)])
        inp["points"] = pts
        if args.point_labels:
            inp["point_labels"] = [int(x) for x in args.point_labels.split(",")]
        if args.mode == "center":
            inp["mode"] = "combined"
    if args.normalized:
        inp["box_normalized"] = True
        inp["normalized"] = True
    if args.keep_canvas:
        inp["keep_canvas"] = True
        inp["crop"] = False
    inp["feather"] = args.feather
    inp["max_side"] = args.max_side
    if args.multimasks:
        inp["return_multimasks"] = True

    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}

    if args.async_run:
        r = requests.post(f"{base}/run", headers=headers, json={"input": inp}, timeout=60)
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id")
        if not job_id:
            print(job, file=sys.stderr)
            return 1
        t0 = time.time()
        while True:
            s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=60)
            s.raise_for_status()
            body = s.json()
            st = body.get("status")
            if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break
            if time.time() - t0 > 300:
                print("poll timeout", body, file=sys.stderr)
                return 1
            time.sleep(1.2)
        if body.get("status") != "COMPLETED":
            print(body, file=sys.stderr)
            return 1
        out = body.get("output") or {}
    else:
        r = requests.post(f"{base}/runsync", headers=headers, json={"input": inp}, timeout=240)
        r.raise_for_status()
        body = r.json()
        out = body.get("output") or body

    if out.get("results"):
        root, ext = os.path.splitext(args.out)
        n_ok = 0
        for i, item in enumerate(out["results"]):
            if not item.get("ok"):
                print(f"[{i}] FAIL {item.get('error_code')} {item.get('error')}", file=sys.stderr)
                continue
            path = f"{root}_{i:02d}{ext or '.png'}"
            _decode_png(item["cutout_png_b64"], path)
            print(f"wrote {path}  size={item.get('cutout_size')}  score={item.get('score')}  bbox={item.get('bbox')}")
            n_ok += 1
        return 0 if n_ok else 1

    if not out.get("ok"):
        print(json.dumps(out, indent=2) if isinstance(out, dict) else out, file=sys.stderr)
        return 1
    _decode_png(out["cutout_png_b64"], args.out)
    print(
        f"wrote {args.out}  size={out.get('cutout_size')}  score={out.get('score')}  "
        f"bbox={out.get('bbox')}  canvas={out.get('canvas')}  mode={out.get('mode')}"
    )
    if out.get("masks"):
        root, ext = os.path.splitext(args.out)
        for i, m in enumerate(out["masks"]):
            path = f"{root}_mask{i}{ext or '.png'}"
            _decode_png(m["mask_png_b64"], path)
            print(f"  multimask {i} score={m.get('score')} bbox={m.get('bbox')} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
