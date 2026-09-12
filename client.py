#!/usr/bin/env python3
"""Call the SAM 2.1 RunPod endpoint and save a cutout PNG."""
from __future__ import annotations

import argparse
import base64
import os
import sys

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    p.add_argument("--image-url")
    p.add_argument("--image", help="local image path (sent as base64)")
    p.add_argument("--mode", default="center", choices=["center", "auto", "point", "box"])
    p.add_argument("--box", help="x1,y1,x2,y2")
    p.add_argument("--out", default="cutout.png")
    args = p.parse_args()
    if not args.endpoint_id or not args.api_key:
        print("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
        return 2

    inp: dict = {"mode": args.mode}
    if args.image_url:
        inp["image_url"] = args.image_url
    elif args.image:
        with open(args.image, "rb") as f:
            inp["image_b64"] = base64.b64encode(f.read()).decode("ascii")
    else:
        print("Need --image-url or --image", file=sys.stderr)
        return 2
    if args.box:
        inp["box"] = [float(x) for x in args.box.split(",")]
        inp["mode"] = "box"

    url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json={"input": inp},
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    out = body.get("output") or body
    if not out.get("ok"):
        print(body, file=sys.stderr)
        return 1
    raw = base64.b64decode(out["cutout_png_b64"])
    with open(args.out, "wb") as f:
        f.write(raw)
    print(f"wrote {args.out}  size={out.get('cutout_size')}  score={out.get('score')}  mode={out.get('mode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
