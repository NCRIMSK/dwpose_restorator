"""Minimal pose visualization functions - independent of controlnet_aux."""
from typing import Tuple, List, Union, Optional
import numpy as np
import cv2
from .pose_types import Keypoint, BodyResult, PoseResult


def decode_json_as_poses(pose_json: dict) -> Tuple[List[PoseResult], List, int, int]:
    """
    Decode pose JSON to PoseResult objects.
    
    Args:
        pose_json: Dict with 'people', 'canvas_height', 'canvas_width'
    
    Returns:
        (poses, animals, height, width)
    """
    height = pose_json.get("canvas_height", 512)
    width = pose_json.get("canvas_width", 512)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            return Keypoint(x, y, c) if c > 0 else None

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    poses = []
    for pose in pose_json.get("people", []):
        body_keypoints = decompress_keypoints(pose.get("pose_keypoints_2d")) or ([None] * 18)
        poses.append(
            PoseResult(
                body=BodyResult(keypoints=body_keypoints),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
        )
    
    return poses, [], height, width


def draw_poses(
    poses: List[PoseResult],
    H: int,
    W: int,
    draw_body: bool = True,
    draw_hand: bool = True,
    draw_face: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Draw poses on a canvas.
    
    Args:
        poses: List of PoseResult objects
        H: Canvas height
        W: Canvas width
        draw_body: Draw body keypoints
        draw_hand: Draw hand keypoints
        draw_face: Draw face keypoints
    
    Returns:
        Canvas as numpy array (H, W, 3)
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85],
    ]

    stickwidth = 4

    for pose in poses:
        # Draw body
        if draw_body and pose.body.keypoints:
            keypoints = pose.body.keypoints
            for (k1_idx, k2_idx), color in zip(limbSeq, colors):
                k1, k2 = keypoints[k1_idx - 1], keypoints[k2_idx - 1]
                if k1 is not None and k2 is not None:
                    x1, y1 = int(k1.x), int(k1.y)
                    x2, y2 = int(k2.x), int(k2.y)
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        cv2.line(canvas, (x1, y1), (x2, y2), color, stickwidth)

            # Draw joints
            for keypoint, color in zip(keypoints, colors):
                if keypoint is not None:
                    x, y = int(keypoint.x), int(keypoint.y)
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(canvas, (x, y), 3, color, -1)

        # Draw hands
        if draw_hand:
            for hand in [pose.left_hand, pose.right_hand]:
                if hand:
                    _draw_hand_or_face(canvas, hand, W, H)

        # Draw face
        if draw_face and pose.face:
            _draw_hand_or_face(canvas, pose.face, W, H)

    return canvas


def _draw_hand_or_face(canvas: np.ndarray, keypoints: List[Optional[Keypoint]], W: int, H: int):
    """Draw hand or face keypoints."""
    if not keypoints:
        return
    
    color = [0, 255, 255]  # Cyan for hands/faces
    for keypoint in keypoints:
        if keypoint is not None:
            x, y = int(keypoint.x), int(keypoint.y)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 2, color, -1)
