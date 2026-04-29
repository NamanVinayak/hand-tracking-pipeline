"""
HaMeR Rerun Visualizer v4 - Apple EgoDex Style
3D visualization + 2D video overlay with Apple EgoDex inspired minimal aesthetic.

Features:
- Softer colors: light blue (left), soft orange (right)
- Minimal rendering: smaller dots, thinner lines
- Clean, professional look matching Apple research standards

Usage:
    python hamer_visualizer.py 04_hamer_30fps.json --video videos/04.mp4
"""

import json
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from typing import Optional, List, Dict

# MANO skeleton
JOINT_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

# Apple EgoDex style: gradient from lime green to yellow for joints
# Each joint gets a different color based on depth/position
def get_joint_colors_apple_style(joints_3d: np.ndarray, side: str) -> list:
    """
    Generate Apple EgoDex style gradient colors for joints.
    Returns list of RGB colors, one per joint.
    Gradient from lime green [100, 255, 100] to yellow [255, 255, 100]
    """
    # Use Z-depth for color gradient (closer = greener, farther = yellower)
    z_values = joints_3d[:, 2]
    z_min, z_max = z_values.min(), z_values.max()

    colors = []
    for z in z_values:
        # Normalize Z to 0-1 range
        if z_max != z_min:
            t = (z - z_min) / (z_max - z_min)
        else:
            t = 0.5

        # Gradient from lime green to yellow
        # Lime green [100, 255, 100] -> Yellow [255, 255, 100]
        r = int(100 + t * 155)  # 100 -> 255
        g = 255                  # constant
        b = 100                  # constant

        colors.append([r, g, b])

    return colors

# Line colors: blue for skeleton (Apple style)
SKELETON_LINE_COLOR_LEFT = [50, 150, 255]   # Blue for left hand skeleton
SKELETON_LINE_COLOR_RIGHT = [255, 120, 50]  # Orange for right hand skeleton


def load_results(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def transform_joints_for_3d(hand: dict, img_width: int, img_height: int, side: str) -> np.ndarray:
    """
    Transform HaMeR 3D joints to Rerun coordinate space.
    Uses ViTPose 2D keypoints for XY positioning (same source as perfect 2D overlay).

    Args:
        hand: Hand data dict with joints_3d, vitpose_2d, bbox
        img_width: Image width in pixels
        img_height: Image height in pixels
        side: "left" or "right" - anatomical hand side from HaMeR detection
    """
    joints_3d = np.array(hand["joints_3d"])

    # Get ViTPose 2D keypoints (same data that makes 2D overlay perfect!)
    if "vitpose_2d" in hand:
        vitpose_2d = np.array(hand["vitpose_2d"])  # (21, 2) in pixel coords
    else:
        # Fallback to bbox if vitpose_2d not available
        bbox = hand["bbox"]
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        vitpose_2d = np.array([[bbox_center_x, bbox_center_y]] * 21)

    # Calculate wrist position (joint 0) in normalized coordinates
    wrist_2d = vitpose_2d[0]  # WRIST is index 0
    norm_x = (wrist_2d[0] - img_width/2) / (img_width/2)
    norm_y = (wrist_2d[1] - img_height/2) / (img_width/2)

    # Keep hand shape from joints_3d (preserve HaMeR's accurate hand geometry)
    scale = 2.0
    result = joints_3d * scale

    # UNIVERSAL SOLUTION: Flip only RIGHT hands (HaMeR canonical space convention)
    # HaMeR outputs left hands correctly, but right hands need X-axis flip
    if side == "right":
        result[:, 0] = -result[:, 0]

    # Position with FULL scale (1.0) to track video positions exactly
    # This ensures hands move together/apart in 3D matching video movement
    # FIXED: Negate X to convert from image coords to Rerun 3D coords
    result[:, 0] -= norm_x * 1.0  # Right in video → appears right in 3D
    result[:, 1] += -norm_y * 1.0  # Flip Y for Rerun coords
    result[:, 2] = -result[:, 2]    # Flip Z for Rerun coords

    return result


def log_hand_3d(hand_key: str, positions: np.ndarray, side: str, joints_3d_original: np.ndarray):
    """Log 3D hand skeleton in Apple EgoDex style with gradient joint colors."""
    # Get Apple-style gradient colors for joints (lime green -> yellow)
    joint_colors = get_joint_colors_apple_style(joints_3d_original, side)

    # Apple EgoDex style: smaller, more minimal dots with gradient colors
    radii = [0.006 if JOINT_NAMES[i].endswith("TIP") else 0.004 for i in range(21)]
    rr.log(
        f"world/{hand_key}/joints",
        rr.Points3D(positions, colors=joint_colors, radii=radii)
    )

    # Apple EgoDex style: blue skeleton lines (not gradient)
    skeleton_color = SKELETON_LINE_COLOR_LEFT if side == "left" else SKELETON_LINE_COLOR_RIGHT
    segments = [[positions[s], positions[e]] for s, e in SKELETON_CONNECTIONS
                if s < len(positions) and e < len(positions)]
    rr.log(
        f"world/{hand_key}/skeleton",
        rr.LineStrips3D(segments, colors=[skeleton_color], radii=0.003)
    )


def project_joints_with_camera(joints_3d: np.ndarray,
                               camera_t: np.ndarray,
                               focal_length: float,
                               img_size: tuple) -> np.ndarray:
    """
    Project 3D joints to 2D using proper perspective projection.

    Args:
        joints_3d: (21, 3) array of 3D joint positions
        camera_t: (3,) array of camera translation [tx, ty, tz]
        focal_length: Focal length in pixels
        img_size: (width, height) of image

    Returns:
        (21, 2) array of 2D pixel coordinates
    """
    # Apply camera translation to get joints in camera space
    joints_cam = joints_3d + camera_t

    # Perspective projection: x_2d = f * x / z + cx
    img_w, img_h = img_size
    cx, cy = img_w / 2, img_h / 2

    x_2d = (focal_length * joints_cam[:, 0] / joints_cam[:, 2]) + cx
    y_2d = (focal_length * joints_cam[:, 1] / joints_cam[:, 2]) + cy

    return np.stack([x_2d, y_2d], axis=1)


def get_joints_2d(hand: dict, img_width: int = 1920, img_height: int = 1080) -> np.ndarray:
    """
    Get 2D joint coordinates.
    Priority: ViTPose keypoints > Camera projection > Bbox projection
    """
    # FIRST: Use ViTPose keypoints if available (most accurate, already in pixel coords!)
    if "vitpose_2d" in hand:
        return np.array(hand["vitpose_2d"])

    joints_3d = np.array(hand["joints_3d"])
    camera_t = np.array(hand["camera_t"])
    bbox = hand["bbox"]

    # SECOND: Check if camera_t depth is valid (not the broken z=39m case)
    # Valid depth should be < 5 meters for hand tracking
    if camera_t[2] > 0.1 and camera_t[2] < 5.0:
        # Use proper perspective projection
        focal_length = 5000.0  # HaMeR's default focal length
        try:
            pts_2d = project_joints_with_camera(
                joints_3d,
                camera_t,
                focal_length,
                (img_width, img_height)
            )
            # Validate results are within reasonable bounds
            if np.all(pts_2d[:, 0] >= -img_width) and np.all(pts_2d[:, 0] <= 2*img_width) and \
               np.all(pts_2d[:, 1] >= -img_height) and np.all(pts_2d[:, 1] <= 2*img_height):
                return pts_2d
        except:
            pass  # Fall through to bbox projection

    # Fallback: bbox projection (for when camera_t is invalid)
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    # Get XY from 3D joints
    xy = joints_3d[:, :2].copy()

    # Scale to fit bbox size
    xy_range = np.ptp(xy, axis=0)  # range (max - min)
    xy_range[xy_range == 0] = 1

    # Scale factor to fit in bbox (with padding)
    scale_x = bbox_w * 0.9 / xy_range[0]
    scale_y = bbox_h * 0.9 / xy_range[1]
    scale = min(scale_x, scale_y)  # uniform scale to preserve shape

    # Center the hand shape
    xy_center = xy.mean(axis=0)
    xy_centered = (xy - xy_center) * scale

    # Project to bbox center
    pts = np.zeros((len(joints_3d), 2))
    pts[:, 0] = bbox_center_x + xy_centered[:, 0]
    pts[:, 1] = bbox_center_y - xy_centered[:, 1]  # flip Y for image coords

    return pts


def draw_arm_2d(frame: np.ndarray, arm_keypoints: dict, side: str):
    """Draw arm skeleton on frame: shoulder → elbow → wrist in Apple EgoDex style."""
    # Use Apple-style colors: blue for left, orange for right
    color = SKELETON_LINE_COLOR_LEFT if side == "left" else SKELETON_LINE_COLOR_RIGHT
    color_bgr = tuple(reversed(color))

    # Extract keypoints (each is [x, y, confidence])
    shoulder = np.array(arm_keypoints["shoulder"][:2])  # Just x, y
    elbow = np.array(arm_keypoints["elbow"][:2])
    wrist = np.array(arm_keypoints["wrist"][:2])

    # Check confidence (index 2) before drawing
    shoulder_conf = arm_keypoints["shoulder"][2]
    elbow_conf = arm_keypoints["elbow"][2]
    wrist_conf = arm_keypoints["wrist"][2]

    # Draw skeleton lines (thin, 1px like hands) - only if confidence > 0.5
    if shoulder_conf > 0.5 and elbow_conf > 0.5:
        cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), color_bgr, 1, cv2.LINE_AA)
    if elbow_conf > 0.5 and wrist_conf > 0.5:
        cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), color_bgr, 1, cv2.LINE_AA)

    # Draw joint dots (3px for arms, slightly smaller than hand tips)
    if shoulder_conf > 0.5:
        cv2.circle(frame, tuple(shoulder.astype(int)), 3, color_bgr, -1, cv2.LINE_AA)
    if elbow_conf > 0.5:
        cv2.circle(frame, tuple(elbow.astype(int)), 3, color_bgr, -1, cv2.LINE_AA)
    # Note: wrist is drawn by hand skeleton, so we skip it here


def draw_hand_2d(frame: np.ndarray, pts: np.ndarray, side: str, joints_3d: np.ndarray):
    """Draw hand skeleton on frame with Apple EgoDex styling and gradient colors."""
    # Get Apple-style gradient colors for joints
    joint_colors_rgb = get_joint_colors_apple_style(joints_3d, side)

    # Skeleton line color (blue/orange based on side)
    skeleton_line_color = SKELETON_LINE_COLOR_LEFT if side == "left" else SKELETON_LINE_COLOR_RIGHT
    line_color_bgr = tuple(reversed(skeleton_line_color))  # RGB -> BGR

    # Apple EgoDex style: thinner lines for minimal aesthetic
    for i, j in SKELETON_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), line_color_bgr, 1, cv2.LINE_AA)

    # Apple EgoDex style: gradient colored dots (lime green -> yellow)
    for i, pt in enumerate(pts):
        r = 4 if JOINT_NAMES[i].endswith("TIP") else 3
        # Convert RGB to BGR for OpenCV
        color_bgr = tuple(reversed(joint_colors_rgb[i]))
        cv2.circle(frame, tuple(pt.astype(int)), r, color_bgr, -1, cv2.LINE_AA)


def visualize(results: dict, video_path: str = None, save_path: str = None):
    """Main visualization."""
    rr.init("HaMeR Hand Tracking", spawn=(save_path is None))
    
    # Layout like MediaPipe visualizer
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D Hands", origin="world"),
        rrb.Vertical(
            rrb.Spatial2DView(name="Video", origin="video"),
            rrb.TextDocumentView(name="Info", origin="info"),
        ),
        column_shares=[2, 1]
    )
    rr.send_blueprint(blueprint)
    
    frames = results.get("frames", [])
    total_frames = len(frames)
    hands_total = sum(len(f.get("hands", [])) for f in frames)
    
    rr.log("info/stats", rr.TextDocument(
        f"**Frames:** {total_frames}\n**Hands:** {hands_total}\n**Joints:** MANO-21",
        media_type=rr.MediaType.MARKDOWN
    ), static=True)
    
    cap = None
    img_w, img_h = 640, 480
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Visualizing {len(frames)} frames...")
    
    for idx, frame_data in enumerate(frames):
        frame_idx = frame_data["frame_idx"]
        ts = frame_data.get("timestamp_ms", frame_idx * 33.33)
        hands = frame_data.get("hands", [])
        
        rr.set_time("time", timestamp=ts / 1000.0)
        rr.set_time("frame", sequence=frame_idx)
        
        # Get video frame
        frame_rgb = None
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process each hand
        for hand in hands:
            side = hand["side"]
            joints_3d = np.array(hand["joints_3d"])
            bbox = hand["bbox"]

            # 3D visualization with Apple-style gradient colors
            pos_3d = transform_joints_for_3d(hand, img_w, img_h, side)
            log_hand_3d(side, pos_3d, side, joints_3d)  # Pass original joints_3d for color gradient

            # 2D overlay
            if frame_rgb is not None:
                # Draw arm skeleton first (so hand overlays on top)
                if "arm_keypoints_2d" in hand:
                    draw_arm_2d(frame_rgb, hand["arm_keypoints_2d"], side)

                # Then draw hand skeleton
                pts_2d = get_joints_2d(hand, img_w, img_h)
                draw_hand_2d(frame_rgb, pts_2d, side, joints_3d)  # Pass joints_3d for gradient colors
        
        if frame_rgb is not None:
            rr.log("video/frame", rr.Image(frame_rgb))
        
        if idx % 50 == 0:
            print(f"  Frame {idx}/{len(frames)}")
    
    if cap:
        cap.release()
    
    if save_path:
        rr.save(save_path)
        print(f"✅ Saved to: {save_path}")
        print(f"   Open: rerun {save_path}")
    else:
        print("✅ Done!")


def visualize_comparison(results_a: dict, results_b: dict, video_path: str = None,
                        label_a: str = "5fps", label_b: str = "30fps"):
    """Compare two results side-by-side."""
    rr.init("HaMeR Comparison", spawn=True)
    
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(name=label_a, origin=f"video/{label_a}"),
        rrb.Spatial2DView(name=label_b, origin=f"video/{label_b}"),
    )
    rr.send_blueprint(blueprint)
    
    frames_a = {f.get("timestamp_ms", f["frame_idx"]*33.33): f for f in results_a.get("frames", [])}
    frames_b = {f.get("timestamp_ms", f["frame_idx"]*33.33): f for f in results_b.get("frames", [])}
    all_ts = sorted(set(frames_a.keys()) | set(frames_b.keys()))
    
    cap = None
    fps = 30
    img_w, img_h = 1920, 1080  # default resolution
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for ts in all_ts:
        rr.set_time("time", timestamp=ts / 1000.0)

        if cap:
            frame_idx = int(ts / 1000.0 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # A
                fa = frame_rgb.copy()
                if ts in frames_a:
                    for h in frames_a[ts].get("hands", []):
                        pts = get_joints_2d(h, img_w, img_h)
                        draw_hand_2d(fa, pts, h["side"])
                rr.log(f"video/{label_a}/frame", rr.Image(fa))

                # B
                fb = frame_rgb.copy()
                if ts in frames_b:
                    for h in frames_b[ts].get("hands", []):
                        pts = get_joints_2d(h, img_w, img_h)
                        draw_hand_2d(fb, pts, h["side"])
                rr.log(f"video/{label_b}/frame", rr.Image(fb))
    
    if cap:
        cap.release()
    print("✅ Comparison done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HaMeR Rerun Visualizer")
    parser.add_argument("json_files", nargs="+", help="HaMeR JSON (1 or 2 files)")
    parser.add_argument("--video", "-v", help="Video for overlay")
    parser.add_argument("--save", "-s", help="Save to .rrd")
    parser.add_argument("--labels", "-l", nargs=2, default=["5fps", "30fps"])
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HaMeR Rerun Visualizer v3")
    print("=" * 50)
    
    if len(args.json_files) == 1:
        results = load_results(args.json_files[0])
        visualize(results, video_path=args.video, save_path=args.save)
    elif len(args.json_files) == 2:
        results_a = load_results(args.json_files[0])
        results_b = load_results(args.json_files[1])
        visualize_comparison(results_a, results_b, video_path=args.video,
                           label_a=args.labels[0], label_b=args.labels[1])
