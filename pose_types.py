"""Minimal pose data structures for visualization."""
from typing import NamedTuple, List, Optional


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


class BodyResult(NamedTuple):
    keypoints: List[Optional[Keypoint]]
    total_score: float = 0.0
    total_parts: int = 0


HandResult = List[Keypoint]
FaceResult = List[Keypoint]


class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Optional[HandResult]
    right_hand: Optional[HandResult]
    face: Optional[FaceResult]
