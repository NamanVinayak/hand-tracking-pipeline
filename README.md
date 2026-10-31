# Hand Tracking Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-supported-2496ED)
![RunPod](https://img.shields.io/badge/RunPod-serverless-8A2BE2)

🖐️ Two-backend hand tracking pipeline for robotics data collection. **MediaPipe** runs locally for fast iteration; **HaMeR** runs on RunPod serverless GPUs for production-grade 3D hand mesh recovery.

> **Status: proof of concept.** Both backends run and produce hand-tracking output on sample egocentric video. It is not a production data pipeline: it has not been run at scale, and error handling and tests are minimal.

---

## Why this exists

Robotics teams collecting human-demonstration data for learning-from-demonstration (LfD) workflows — ALOHA, OpenVLA, ACT — need structured 3D hand pose data extracted from egocentric (first-person) video. Public datasets are limited, so most teams record their own. This pipeline turns raw egocentric video into MANO-21 standard hand-pose JSON ready to feed into a training run.

The two backends sit behind one consistent JSON output schema, so downstream code doesn't care which one ran.

---

## Two backends, one pipeline

| Backend | Where it runs | Model | Use it for |
|---|---|---|---|
| **MediaPipe Hand Landmarker** | Local (CPU/GPU) | 21 keypoints, 2D + relative 3D | Fast iteration, debugging, smaller datasets |
| **HaMeR** | RunPod serverless GPU | Full MANO hand mesh (778 vertices) + 21 joints | Production datasets, accurate 3D ground truth |

**HaMeR upstream:** [geopavlakos/hamer](https://github.com/geopavlakos/hamer) — install separately following their setup. We ship a serverless wrapper around it (`hamer_runpod/`), not the model itself.

---

## Features

- **Egocentric video support** — Optimized for first-person camera framing common in robotics data collection
- **MANO-21 standard output** — 21 named joints (WRIST → THUMB_TIP, INDEX_TIP, etc.) compatible with most LfD frameworks
- **Two-handed tracking** — Left + right hands tracked independently per frame
- **Serverless GPU offload** — No on-prem GPU required; HaMeR runs on RunPod and pulls inputs/outputs from S3 / Cloudflare R2
- **Rerun.io visualizer** — Skeleton overlay viewer in Apple-EgoDex style (small green joints, blue bones)
- **Batch processing** — Process a whole directory of videos with one command
- **Dockerized** — Reproducible serverless container via the included Dockerfiles
- **Strategic roadmap** — `hamer_runpod/ROADMAP.md` maps out the production-readiness path: hand-object interaction tracking, 6D object pose, contact-point detection, and competitive positioning vs commercial alternatives

---

## Quick start (local MediaPipe path)

```bash
# 1. Clone
git clone https://github.com/NamanVinayak/hand-tracking-pipeline.git
cd hand-tracking-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Track a single video
python hand_tracker.py --input demo.mp4 --output demo_landmarks.json

# 4. (Optional) Visualize with Rerun
python rerun_visualizer.py --input demo.mp4 --landmarks demo_landmarks.json
```

The MediaPipe model (`hand_landmarker.task`, ~12MB) is downloaded automatically on first run.

---

## Batch processing

Process every video in a directory:

```bash
python process_videos.py \
  --input-dir ./recordings \
  --output-dir ./tracked \
  --visualize  # optional: writes annotated mp4s alongside the JSON
```

---

## RunPod serverless (HaMeR path)

The serverless container in `hamer_runpod/` packages HaMeR (CUDA 12.1 + PyTorch 2.2.0) with the model weights baked in. It runs on RunPod, reading videos from a Cloudflare R2 / S3 bucket and writing results back.

### One-time setup

1. Set up a Cloudflare R2 (or S3) bucket
2. Build and push the Docker image:
   ```bash
   cd hamer_runpod
   ./build_and_push.sh   # builds and pushes to your container registry
   ```
3. Create a RunPod Serverless endpoint pointing at your image
4. Set credentials via environment variables (see below)

### Submit a job

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
export S3_ENDPOINT_URL=...
export S3_ACCESS_KEY_ID=...
export S3_SECRET_ACCESS_KEY=...
export R2_BUCKET_NAME=...

python hamer_runpod/hamer_client.py --input demo.mp4 --output demo_mesh.json
```

See `hamer_runpod/README.md` for full handler details.

---

## Output format

Each frame produces a record like:

```json
{
  "frame": 42,
  "timestamp": 1.4,
  "hands": {
    "Right": {
      "joints": [
        {"name": "WRIST", "x": 0.51, "y": 0.62, "z": 0.0},
        {"name": "THUMB_CMC", "x": 0.49, "y": 0.59, "z": -0.01},
        ...
      ],
      "confidence": 0.94
    },
    "Left": null
  }
}
```

- `x`, `y` are normalized image coordinates (0–1)
- `z` is relative depth (negative = closer to camera)
- `null` means the hand was not detected in that frame
- HaMeR responses additionally include the full MANO mesh (778 vertices, 1538 faces)

---

## Repo layout

```
hand-tracking-pipeline/
├── hand_tracker.py          Local MediaPipe pipeline (HandTracker class)
├── process_videos.py        Batch processor for a directory of videos
├── rerun_visualizer.py      Rerun.io 3D skeleton viewer
├── requirements.txt
└── hamer_runpod/
    ├── handler.py           RunPod serverless handler (HaMeR inference)
    ├── hamer_client.py      Local client for submitting jobs
    ├── hamer_pipeline.py    Pipeline orchestration
    ├── hamer_visualizer.py  HaMeR-specific mesh visualizer
    ├── Dockerfile           Production serverless image
    ├── Dockerfile.base      Base layer (HaMeR + dependencies)
    ├── build*.sh            Build/push helpers
    ├── README.md            HaMeR-side build & deploy notes
    └── ROADMAP.md           Strategic roadmap toward production
```

---

## Required external assets (not bundled)

These have restrictive licensing and are NOT included in this repo. Set them up separately:

- **MANO hand model** (`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`) — register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) and place under `hamer_runpod/mano_models/`
- **HaMeR upstream code** — clone [geopavlakos/hamer](https://github.com/geopavlakos/hamer) into the Docker build context if you're rebuilding the serverless image
- **HaMeR model weights** — downloaded by HaMeR's `fetch_demo_data.sh` during the Docker build

The MediaPipe path needs none of these.

---

## Use cases

- **Robot Learning from Demonstration (LfD)** — Capture hand poses from human operators to build manipulation training datasets
- **Teleoperation data collection** — Extract structured hand mesh data from egocentric video recordings
- **Dexterous manipulation research** — Generate 3D hand pose annotations at scale without a GPU cluster
- **Pose-conditioned generative models** — Hand pose tokens for text-to-video / image-conditioning pipelines

---

## Roadmap

`hamer_runpod/ROADMAP.md` lays out the production path:

- ⭐⭐⭐⭐⭐ Hand-object interaction tracking (object IDs, contact points, 6D pose)
- ⭐⭐⭐⭐ Multi-camera fusion for occluded hands
- ⭐⭐⭐ Real-time streaming inference (currently batch-only)
- ⭐⭐⭐ Action segmentation (per-clip task labels)
- ⭐⭐ Cost optimization (warm workers, batched submission)

Read the full roadmap for the competitive landscape vs commercial alternatives and the priority order for a robotics-data startup.

---

## Troubleshooting

**No hands detected in any frame**
- MediaPipe is sensitive to motion blur and unusual angles. Try a clearer test clip first.
- For egocentric (helmet-cam) footage, hands sometimes leave the frame entirely — check the source video.

**RunPod job times out**
- Large videos can exceed the default serverless timeout. Split into <60s chunks before submitting.
- Verify `RUNPOD_API_KEY` is set and the endpoint URL points to your HaMeR-deployed endpoint.
- Cold starts can take 60–90s for the HaMeR image — first request after idle will be slow.

**Rerun visualizer is blank**
- Make sure `rerun-sdk` is installed (`pip install rerun-sdk`) and your terminal supports the `rerun` command.
- The viewer expects a JSON file produced by `hand_tracker.py` — passing raw video alone won't work.

---

## Contributing

PRs welcome. Areas where help is most useful:
- Smoothing/filtering of jittery landmark traces
- More backends (e.g. OpenPose, MMPose) behind the same JSON schema
- Better defaults for low-light / fast-motion robotics footage
- Streaming inference (currently all jobs are batch)

---

## License

MIT — see [LICENSE](LICENSE).

The HaMeR model and MANO models retain their respective licenses; this repo only ships the wrapper code and pipeline orchestration.
