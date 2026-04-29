# HaMeR RunPod Serverless Endpoint

Hand Mesh Recovery (HaMeR) as a serverless API on RunPod.

## What's Included

- **Dockerfile**: CUDA 12.1 + PyTorch 2.2.0 + HaMeR + 6GB models baked in
- **handler.py**: RunPod serverless handler with S3 integration

## Build & Deploy

### 1. Build Docker Image

```bash
docker build -t yourusername/hamer-runpod:latest .
```

### 2. Push to Docker Hub

```bash
docker login
docker push yourusername/hamer-runpod:latest
```

### 3. Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Click "New Endpoint"
3. Select "Custom Container"
4. Enter: `yourusername/hamer-runpod:latest`
5. Configure GPU (A40 recommended)
6. Add environment secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`

## API Usage

```python
import runpod

runpod.api_key = "your_runpod_api_key"

result = runpod.run_sync(
    endpoint_id="your_endpoint_id",
    input={
        "s3_url": "s3://your-bucket/video.mp4",
        "fps": 5  # sample rate
    }
)

# Returns:
# {
#   "frames": [
#     {
#       "frame_idx": 0,
#       "timestamp_ms": 0.0,
#       "hands": [
#         {
#           "side": "left",
#           "joints_3d": [[x,y,z], ...],  # 21 joints
#           "camera_t": [tx, ty, tz],
#           "bbox": [x1, y1, x2, y2]
#         }
#       ]
#     }
#   ]
# }
```

## Output Format

Each hand returns 21 3D joints (MANO format):
- WRIST
- THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP
- INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
- MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP
- RING_MCP, RING_PIP, RING_DIP, RING_TIP
- PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
