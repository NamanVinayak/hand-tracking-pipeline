#!/usr/bin/env python3
"""
HaMeR Pipeline - End-to-end video processing and visualization.

Process any video through HaMeR on RunPod and visualize results in Rerun.

Usage:
    # Process and visualize
    python hamer_pipeline.py videos/my_video.mp4 --fps 30 --visualize
    
    # Just process
    python hamer_pipeline.py videos/my_video.mp4 --fps 30
    
    # Compare different FPS settings
    python hamer_pipeline.py videos/my_video.mp4 --compare 5 30
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def upload_to_r2(video_path: str) -> str:
    """Upload video to Cloudflare R2 and return storage URL."""
    import boto3
    
    video_path = Path(video_path)
    
    s3_client = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('S3_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('S3_REGION', 'auto')
    )
    
    bucket = "egocentric-videos"
    key = f"videos/{video_path.name}"
    
    print(f"Uploading {video_path.name} to R2...")
    s3_client.upload_file(str(video_path), bucket, key)
    
    storage_url = f"s3://{bucket}/{key}"
    print(f"Uploaded to: {storage_url}")
    return storage_url


def process_video(storage_url: str, fps: int = 30, timeout: int = 600) -> dict:
    """Process video on RunPod and return results."""
    import runpod
    
    runpod.api_key = os.environ.get('RUNPOD_API_KEY')
    endpoint_id = os.environ.get('RUNPOD_ENDPOINT_ID', 'foah778z6qr2zd')
    
    endpoint = runpod.Endpoint(endpoint_id)
    
    print(f"Processing with HaMeR @ {fps} FPS...")
    run = endpoint.run({
        "storage_url": storage_url,
        "fps": fps
    })
    
    print(f"Job ID: {run.job_id}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = run.status()
        elapsed = int(time.time() - start_time)
        print(f"\r  Status: {status} ({elapsed}s elapsed)", end="", flush=True)
        
        if status == "COMPLETED":
            print(f"\n✅ Done!")
            return run.output()
        elif status == "FAILED":
            print(f"\n❌ Failed!")
            raise Exception(f"Job failed: {run.output()}")
        
        time.sleep(5)
    
    raise TimeoutError(f"Job timed out after {timeout}s")


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def visualize(json_path: str, video_path: str = None):
    """Launch Rerun visualization."""
    from hamer_visualizer import load_results, visualize_single
    
    results = load_results(json_path)
    visualize_single(results, video_path=video_path)


def compare(json_path_a: str, json_path_b: str, video_path: str = None,
            label_a: str = "5fps", label_b: str = "30fps"):
    """Launch Rerun comparison visualization."""
    from hamer_visualizer import load_results, visualize_comparison
    
    results_a = load_results(json_path_a)
    results_b = load_results(json_path_b)
    visualize_comparison(results_a, results_b, video_path=video_path,
                        label_a=label_a, label_b=label_b)


def main():
    parser = argparse.ArgumentParser(description="HaMeR Pipeline - Process and Visualize")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second to sample (default: 30)")
    parser.add_argument("--output", "-o", help="Output JSON path (default: <video>_hamer.json)")
    parser.add_argument("--visualize", "-V", action="store_true", help="Launch Rerun visualization")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("FPS1", "FPS2"),
                        help="Compare two FPS settings (e.g., --compare 5 30)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds (default: 600)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("HaMeR Pipeline")
    print("=" * 50)
    
    # Upload video once
    storage_url = upload_to_r2(str(video_path))
    
    if args.compare:
        # Compare mode: process at two different FPS settings
        fps_a, fps_b = args.compare
        
        print(f"\n--- Processing at {fps_a} FPS ---")
        results_a = process_video(storage_url, fps=fps_a, timeout=args.timeout)
        output_a = video_path.stem + f"_hamer_{fps_a}fps.json"
        save_results(results_a, output_a)
        
        print(f"\n--- Processing at {fps_b} FPS ---")
        results_b = process_video(storage_url, fps=fps_b, timeout=args.timeout)
        output_b = video_path.stem + f"_hamer_{fps_b}fps.json"
        save_results(results_b, output_b)
        
        print("\n" + "=" * 50)
        print("COMPARISON COMPLETE")
        print("=" * 50)
        print(f"  {fps_a} FPS: {results_a.get('total_frames', 0)} frames")
        print(f"  {fps_b} FPS: {results_b.get('total_frames', 0)} frames")
        
        if args.visualize:
            compare(output_a, output_b, video_path=str(video_path),
                   label_a=f"{fps_a}fps", label_b=f"{fps_b}fps")
    else:
        # Single mode
        results = process_video(storage_url, fps=args.fps, timeout=args.timeout)
        
        output_path = args.output or (video_path.stem + "_hamer.json")
        save_results(results, output_path)
        
        total = results.get("total_frames", 0)
        print("\n" + "=" * 50)
        print(f"COMPLETE: {total} frames processed")
        print("=" * 50)
        
        if args.visualize:
            visualize(output_path, video_path=str(video_path))


if __name__ == "__main__":
    main()
