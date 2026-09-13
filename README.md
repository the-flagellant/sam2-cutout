# SAM 2.1 RunPod serverless worker v2

Same image-in / PNG-out contract as v1, plus the controls we kept reinventing around the old worker.

## What changed
- Soft alpha (distance feather) + fringe decontamination
- Hole-fill, morph close, keep-largest connected component
- `keep_canvas` / `crop=false` + `bbox` / `bbox_norm` in source pixels
- Box **and** points (pos + neg) in one call (`mode=combined` or just pass both)
- `return_multimasks` — do not silently keep argmax
- Batch: `images: [{...}, ...]`
- EXIF transpose + optional `rotate`
- `max_side` downsample → upsample mask
- 0–1 normalized boxes/points (`box_normalized`)
- `prefer_standing` for auto / pick among multimasks
- `/run` + poll supported by `client.py --async`
- Structured `{ok, error_code, error}` instead of a mystery blob

Old clients still work: `image_url` + `mode=center` still returns `cutout_png_b64`.

## Files
- handler.py
- Dockerfile
- requirements.txt
- client.py
- test_input.json

## Build
```bash
docker build --platform linux/amd64 -t YOURUSER/sam2-cutout:2.0 .
docker push YOURUSER/sam2-cutout:2.0
```

## Deploy
RunPod console → Serverless → New Endpoint → Import from Docker Registry

| Setting | Value |
|---|---|
| Image | docker.io/YOURUSER/sam2-cutout:2.0 |
| Type | Queue |
| GPU | 16–24 GB |
| Active workers | 0 (or 1 if you scrape often) |
| Max workers | 2 |
| Idle timeout | 30–60 s recommended |
| Execution timeout | 180–300 s |
| FlashBoot | on |
| Container disk | ≥ 20 GB |
| Env | `SAM2_SIZE=large` |

URL: `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

## Call
```bash
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_API_KEY=...

# standing portrait, soft matte, keep full canvas
python client.py --image-url 'https://example.com/pose.jpg' \
  --mode center --keep-canvas --feather 5 --out tank.png

# box + negative point (locker / truck)
python client.py --image shot.jpg --mode combined \
  --box 0.15,0.05,0.85,0.98 --normalized \
  --points 0.5,0.45;0.08,0.80 --point-labels 1,0 \
  --out kyle.png

# batch pose sheet
python client.py --images a.jpg b.jpg c.jpg --mode auto --out sheet.png

```

## Input (single)
| field | default | notes |
|---|---|---|
| image_url / image_b64 | required | URL fetch retries + browser UA |
| mode | center | auto, center, point, box, combined |
| box | — | `[x1,y1,x2,y2]` px or 0–1 |
| points / point_labels | — | `1` foreground, `0` background |
| box_normalized | false | also auto-detected if all values ≤ 1.5 |
| crop / keep_canvas | crop=true | full-frame RGBA + bbox when keep_canvas |
| feather | 4 | px inner fade |
| close / erode | 3 / 0 | morph |
| hole_fill / keep_largest | true | |
| pad_pct | 0 | pad bbox |
| max_side | 1600 | 0 = full res |
| prefer_standing | true | tall instance bias |
| return_multimasks / multimask_max | false / 3 | |
| rotate | 0 | 90/180/270 after EXIF |
| images | — | batch of the same fields |

## Output
`ok, version, mode, model, image_size, cutout_size, score, bbox, bbox_norm, canvas, cutout_png_b64, mask_png_b64, masks?`

Batch: `{ok, count, ok_count, results:[...]}`.
