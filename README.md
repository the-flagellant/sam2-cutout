# SAM 2.1 RunPod serverless worker

Replaces local U²-Net-P cutouts with Meta SAM 2.1 Hiera-Large.

## Files
- handler.py
- Dockerfile
- requirements.txt
- client.py
- test_input.json (optional)

## Build
```bash
docker build --platform linux/amd64 -t YOURUSER/sam2-cutout:1.0 .
docker push YOURUSER/sam2-cutout:1.0
```

## Deploy
RunPod console → Serverless → New Endpoint → Import from Docker Registry

| Setting | Value |
|---|---|
| Image | docker.io/YOURUSER/sam2-cutout:1.0 |
| Type | Queue |
| GPU | 16–24 GB |
| Active workers | 0 |
| Max workers | 2 |
| Idle timeout | 5 s |
| Execution timeout | 120 s |
| FlashBoot | on |
| Container disk | ≥ 20 GB |
| Env | SAM2_SIZE=large |

URL: `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

## Call
```bash
export RUNPOD_ENDPOINT_ID=...
export RUNPOD_API_KEY=...
python client.py --image-url 'https://pbs.twimg.com/media/HRc0NmvXEAA8dai.jpg?name=large' --mode center --out tank_cut.png
```

Modes: center | auto | point | box
Response fields: cutout_png_b64, mask_png_b64, score, cutout_size
