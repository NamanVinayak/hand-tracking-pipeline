"""
Rerun.io Visualization - Apple EgoDex Style
FIXED: Tiny dots, thin lines, skeleton-like appearance.
"""

import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
from hand_tracker import HandTracker, JOINT_NAMES


# Hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

# Apple-style colors: small green/yellow dots, blue lines
DOT_COLOR = [100, 255, 100]  # Lime green
LINE_COLOR = [50, 150, 255]  # Blue


def log_hand_skeleton(hand_key: str, hand_data: Dict[str, Any]):
    """Log hand as thin skeleton matching Apple EgoDex style."""
    if not hand_data or not hand_data.get("detected"):
        return
    
    landmarks = hand_data["landmarks_3d"]
    positions = []
    
    # Use actual video coordinates
    SCALE = 2.0
    
    for lm in landmarks:
        x = (lm["x"] - 0.5) * SCALE
        y = -(lm["y"] - 0.5) * SCALE
        z = -lm["z"] * SCALE
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # TINY dots like Apple (0.006 instead of 0.018)
    radii = [0.008 if JOINT_NAMES[i].endswith("TIP") else 0.006 for i in range(21)]
    
    rr.log(
        f"world/{hand_key}/joints",
        rr.Points3D(positions, colors=[DOT_COLOR] * 21, radii=radii)
    )
    
    # THIN lines like Apple (0.003 instead of 0.025)
    segments = [[positions[s], positions[e]] for s, e in HAND_CONNECTIONS]
    rr.log(
        f"world/{hand_key}/skeleton",
        rr.LineStrips3D(segments, colors=[LINE_COLOR], radii=0.004)
    )


def process_video_to_rerun(
    video_path: str,
    output_rrd_path: Optional[str] = None,
    spawn_viewer: bool = True
):
    video_path = Path(video_path)
    rr.init(f"egodex_{video_path.stem}", spawn=spawn_viewer)
    
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D Hands", origin="world"),
        rrb.Vertical(
            rrb.Spatial2DView(name="Video", origin="video"),
            rrb.TextDocumentView(name="Info", origin="info"),
        ),
        column_shares=[2, 1]
    )
    rr.send_blueprint(blueprint)
    
    rr.log("info/task", rr.TextDocument(
        f"**Task:** {video_path.stem}\n\n**Joints:** MANO-21",
        media_type=rr.MediaType.MARKDOWN
    ), static=True)
    
    # Very low thresholds for maximum detection
    tracker = HandTracker(
        min_detection_confidence=0.1,  # Minimum practical value
        min_tracking_confidence=0.1
    )
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing {video_path.name}: {total_frames} frames @ {fps} FPS")
    
    frame_idx = 0
    left_detected = 0
    right_detected = 0
    both_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = int(frame_idx * 1000 / fps)
        
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_seconds("video_time", timestamp_ms / 1000.0)
        
        hands_data = tracker.process_frame(frame, timestamp_ms, width, height)
        
        rr.log("video/frame", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        
        # Track detection stats
        has_left = hands_data.get("left") is not None
        has_right = hands_data.get("right") is not None
        if has_left:
            left_detected += 1
        if has_right:
            right_detected += 1
        if has_left and has_right:
            both_detected += 1
        
        log_hand_skeleton("left", hands_data.get("left"))
        log_hand_skeleton("right", hands_data.get("right"))
        
        if frame_idx % 30 == 0:
            print(f"\rFrame {frame_idx}/{total_frames}", end="", flush=True)
        
        frame_idx += 1
    
    cap.release()
    tracker.close()
    
    # Print tracking statistics
    print(f"\n\n=== TRACKING STATS ===")
    print(f"Total frames: {frame_idx}")
    print(f"Left hand detected: {left_detected} ({left_detected/frame_idx*100:.1f}%)")
    print(f"Right hand detected: {right_detected} ({right_detected/frame_idx*100:.1f}%)")
    print(f"Both hands detected: {both_detected} ({both_detected/frame_idx*100:.1f}%)")
    print(f"========================\n")
    
    if output_rrd_path:
        rr.save(output_rrd_path)
        print(f"✓ Saved: {output_rrd_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output-rrd", "-o")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    
    process_video_to_rerun(args.input, args.output_rrd, not args.no_viewer)


if __name__ == "__main__":
    main()
