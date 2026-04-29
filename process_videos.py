"""
Batch processing script for hand tracking.
Processes all videos in a directory and outputs JSON tracking data.
"""

import argparse
from pathlib import Path
from hand_tracker import HandTracker


def process_directory(
    input_dir: str,
    output_dir: str,
    generate_visualization: bool = False
):
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save outputs
        generate_visualization: Whether to create visualization videos
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    videos = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"No video files found in: {input_dir}")
        return
    
    print(f"Found {len(videos)} video(s) to process\n")
    
    # Initialize tracker
    tracker = HandTracker()
    
    results_summary = []
    
    for idx, video_path in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Processing: {video_path.name}")
        
        # Output paths
        stem = video_path.stem
        json_path = output_path / f"{stem}_hands.json"
        video_out_path = output_path / f"{stem}_visualized.mp4" if generate_visualization else None
        
        # Progress callback
        def show_progress(current, total):
            pct = current / total * 100
            print(f"\r  Progress: {current}/{total} ({pct:.1f}%)", end="", flush=True)
        
        try:
            result = tracker.process_video(
                str(video_path),
                output_json_path=str(json_path),
                output_video_path=str(video_out_path) if video_out_path else None,
                progress_callback=show_progress
            )
            
            results_summary.append({
                "video": video_path.name,
                "total_frames": result["total_frames"],
                "frames_with_hands": result["frames_with_hands"],
                "detection_rate": result["detection_rate"]
            })
            
            print(f"\n  ✅ Done - Detection rate: {result['detection_rate']}%\n")
            
            # Reset tracker for next video (timestamps must be monotonic per video)
            tracker.reset()
            
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")
            results_summary.append({
                "video": video_path.name,
                "error": str(e)
            })
    
    tracker.close()
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    for r in results_summary:
        if "error" in r:
            print(f"  ❌ {r['video']}: ERROR - {r['error']}")
        else:
            print(f"  ✅ {r['video']}: {r['detection_rate']}% detection rate ({r['frames_with_hands']}/{r['total_frames']} frames)")
    
    print("="*60)
    print(f"\nOutputs saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos for hand tracking"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing videos"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for JSON files (default: output)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Also generate visualization videos"
    )
    
    args = parser.parse_args()
    
    process_directory(
        args.input,
        args.output,
        generate_visualization=args.visualize
    )


if __name__ == "__main__":
    main()
