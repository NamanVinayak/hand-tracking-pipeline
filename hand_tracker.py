"""
Hand and Finger Tracking Module with Industry-Standard Output.
Outputs MANO-21 compatible landmarks with both 2D and 3D coordinates.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# MediaPipe 0.10+ uses task-based API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MANO-21 Standard Joint Names
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Rainbow colors for each finger (for visualization)
FINGER_COLORS = {
    "WRIST": (255, 255, 255),       # White
    "THUMB": (255, 0, 0),           # Red
    "INDEX": (255, 165, 0),         # Orange  
    "MIDDLE": (255, 255, 0),        # Yellow
    "RING": (0, 255, 0),            # Green
    "PINKY": (0, 0, 255),           # Blue
}


def get_joint_color(joint_name: str) -> tuple:
    """Get color for a joint based on its finger."""
    if joint_name == "WRIST":
        return FINGER_COLORS["WRIST"]
    for finger in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
        if joint_name.startswith(finger):
            return FINGER_COLORS[finger]
    return (255, 255, 255)


def download_model_if_needed(model_path: str = "hand_landmarker.task") -> str:
    """Download the hand landmarker model if not present."""
    if not Path(model_path).exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Model saved to: {model_path}")
    return model_path


class HandTracker:
    """
    Processes video frames for hand and finger tracking using MediaPipe 0.10+.
    Outputs industry-standard MANO-21 compatible data.
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = "hand_landmarker.task"
    ):
        """Initialize the hand tracker."""
        model_path = download_model_if_needed(model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
    
    def reset(self):
        """Reset for a new video (required for monotonic timestamps)."""
        self.landmarker.close()
        
        model_path = download_model_if_needed()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp_ms: int,
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Process a single frame and extract hand landmarks.
        Returns MANO-21 compatible format with 2D and 3D coordinates.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_data = {
            "left": None,
            "right": None
        }
        
        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness
            ):
                hand_label = handedness[0].category_name.lower()
                hand_score = handedness[0].score
                
                # Extract landmarks in MANO-21 format
                landmarks_3d = []
                landmarks_2d = []
                
                for idx, landmark in enumerate(hand_landmarks):
                    # 3D coordinates (normalized, camera frame)
                    landmarks_3d.append({
                        "id": idx,
                        "name": JOINT_NAMES[idx],
                        "x": round(landmark.x, 6),
                        "y": round(landmark.y, 6),
                        "z": round(landmark.z, 6)
                    })
                    
                    # 2D pixel coordinates
                    landmarks_2d.append({
                        "id": idx,
                        "name": JOINT_NAMES[idx],
                        "u": int(landmark.x * frame_width),
                        "v": int(landmark.y * frame_height)
                    })
                
                hands_data[hand_label] = {
                    "detected": True,
                    "confidence": round(hand_score, 4),
                    "landmarks_3d": landmarks_3d,
                    "landmarks_2d": landmarks_2d
                }
        
        return hands_data
    
    def draw_landmarks(self, frame: np.ndarray, hands_data: Dict[str, Any]) -> np.ndarray:
        """Draw hand landmarks with rainbow finger colors."""
        annotated = frame.copy()
        
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),    # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for hand_key in ["left", "right"]:
            hand = hands_data.get(hand_key)
            if not hand or not hand.get("detected"):
                continue
            
            landmarks_2d = hand["landmarks_2d"]
            points = [(lm["u"], lm["v"]) for lm in landmarks_2d]
            
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                color = get_joint_color(JOINT_NAMES[end_idx])
                cv2.line(annotated, points[start_idx], points[end_idx], color, 2)
            
            # Draw joints
            for i, (px, py) in enumerate(points):
                color = get_joint_color(JOINT_NAMES[i])
                radius = 6 if JOINT_NAMES[i].endswith("TIP") else 4
                cv2.circle(annotated, (px, py), radius, color, -1)
            
            # Add label
            cv2.putText(
                annotated, 
                f"{hand_key.title()}: {hand['confidence']:.2f}",
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return annotated
    
    def process_video(
        self,
        video_path: str,
        output_json_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process an entire video file.
        Returns industry-standard JSON output.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frames_data = []
        frames_with_hands = 0
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp_ms = int(frame_idx * 1000 / fps)
            hands_data = self.process_frame(frame, timestamp_ms, width, height)
            
            frame_entry = {
                "frame_id": frame_idx,
                "timestamp_ms": timestamp_ms,
                "hands": hands_data
            }
            frames_data.append(frame_entry)
            
            if hands_data["left"] or hands_data["right"]:
                frames_with_hands += 1
            
            if video_writer:
                annotated = self.draw_landmarks(frame, hands_data)
                video_writer.write(annotated)
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)
            
            frame_idx += 1
        
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Industry-standard output format
        output = {
            "metadata": {
                "video_id": Path(video_path).stem,
                "source_file": Path(video_path).name,
                "fps": fps,
                "resolution": [width, height],
                "total_frames": total_frames,
                "frames_with_hands": frames_with_hands,
                "detection_rate": round(frames_with_hands / total_frames * 100, 2) if total_frames > 0 else 0,
                "coordinate_system": "camera_frame",
                "joint_standard": "MANO_21"
            },
            "frames": frames_data
        }
        
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved tracking data to: {output_json_path}")
        
        return output
    
    def close(self):
        """Release resources."""
        self.landmarker.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand tracking with MediaPipe (MANO-21 output)")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output-json", "-j", help="Output JSON path")
    parser.add_argument("--output-video", "-v", help="Output visualized video path")
    args = parser.parse_args()
    
    tracker = HandTracker()
    
    def show_progress(current, total):
        pct = current / total * 100
        print(f"\rProcessing: {current}/{total} frames ({pct:.1f}%)", end="", flush=True)
    
    print(f"Processing: {args.input}")
    result = tracker.process_video(
        args.input,
        output_json_path=args.output_json,
        output_video_path=args.output_video,
        progress_callback=show_progress
    )
    
    print(f"\n\nResults:")
    print(f"  Total frames: {result['metadata']['total_frames']}")
    print(f"  Frames with hands: {result['metadata']['frames_with_hands']}")
    print(f"  Detection rate: {result['metadata']['detection_rate']}%")
    print(f"  Joint standard: {result['metadata']['joint_standard']}")
    
    tracker.close()


if __name__ == "__main__":
    main()
