"""
HaMeR RunPod Client
Upload video to Cloudflare R2, trigger HaMeR processing, get hand coordinates.
"""

import os
import json
import time
import boto3
import runpod
from pathlib import Path
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Configure RunPod
runpod.api_key = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

# Configure R2
R2_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
R2_ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
R2_SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "egocentric-videos")


def get_r2_client():
    """Create R2 client."""
    return boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name='auto'
    )


def upload_video(video_path: str) -> str:
    """Upload video to R2 and return the storage URL."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Use filename as key
    key = f"videos/{video_path.name}"
    
    print(f"Uploading {video_path.name} to R2...")
    client = get_r2_client()
    client.upload_file(str(video_path), R2_BUCKET, key)
    
    storage_url = f"s3://{R2_BUCKET}/{key}"
    print(f"Uploaded to: {storage_url}")
    return storage_url


def process_video(storage_url: str, fps: int = 5, timeout: int = 300) -> dict:
    """
    Trigger HaMeR processing on RunPod and wait for results.
    
    Args:
        storage_url: R2 URL (s3://bucket/path)
        fps: Frames per second to sample
        timeout: Max wait time in seconds
    
    Returns:
        Dict with hand coordinates for each frame
    """
    print(f"Triggering HaMeR on RunPod...")
    print(f"  Endpoint: {ENDPOINT_ID}")
    print(f"  Video: {storage_url}")
    print(f"  FPS: {fps}")
    
    # Create endpoint and run
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    run = endpoint.run({
        "storage_url": storage_url,
        "fps": fps
    })
    
    print(f"Job ID: {run.job_id}")
    print("Waiting for results...")
    
    # Poll for results
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = run.status()
        
        if status == "COMPLETED":
            result = run.output()
            print(f"✅ Done! Processed {result.get('total_frames', 0)} frames")
            return result
        
        elif status == "FAILED":
            error = run.output()
            print(f"❌ Failed: {error}")
            return {"error": error}
        
        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = int(time.time() - start_time)
            print(f"  Status: {status} ({elapsed}s elapsed)")
            time.sleep(5)
        
        else:
            print(f"  Unknown status: {status}")
            time.sleep(5)
    
    return {"error": f"Timeout after {timeout}s"}


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


# ========== Main ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HaMeR RunPod Client")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=int, default=5, help="FPS to sample (default: 5)")
    parser.add_argument("--output", "-o", help="Output JSON path (default: <video>_hamer.json)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        video_stem = Path(args.video).stem
        args.output = f"{video_stem}_hamer.json"
    
    print("=" * 50)
    print("HaMeR RunPod Client")
    print("=" * 50)
    
    # Upload video
    storage_url = upload_video(args.video)
    
    # Process
    results = process_video(storage_url, fps=args.fps, timeout=args.timeout)
    
    # Save
    if "error" not in results:
        save_results(results, args.output)
        
        # Print summary
        frames = results.get("frames", [])
        frames_with_hands = sum(1 for f in frames if f.get("hands"))
        print(f"\nSummary:")
        print(f"  Total frames: {len(frames)}")
        print(f"  Frames with hands: {frames_with_hands}")
        if frames_with_hands > 0:
            print(f"  Detection rate: {100*frames_with_hands/len(frames):.1f}%")
    else:
        print(f"\nError: {results['error']}")
