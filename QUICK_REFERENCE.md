# Quick Reference Card: Relative Restoration

## Node Usage

### ComfyUI Node Setup
```
Input (Required):
  - pose_keypoints: Current pose with some missing keypoints

Input (Optional):
  - ref_pose: Reference pose for restoration (same format)
  - reduce_confidence: [True/False] Reduce confidence of restored points
  - confidence_reduction_factor: [0.0-1.0] Confidence multiplier
  - use_gpu: [True/False] Optional - use GPU if available

Output:
  - pose_image: Visualization of restored pose (RGB tensor)
  - pose_restored: Pose data with restored keypoints
```

## Algorithm at a Glance

```
┌─ Extract keypoints from flat list
├─ Estimate how pose changed (affine transform)
├─ For each missing keypoint:
│  ├─ Find parent in skeleton
│  ├─ Get offset from parent→child in reference
│  ├─ Transform offset using affine
│  └─ Add to parent's position
├─ Zero-out out-of-canvas keypoints for visualization/export
└─ Return restored pose + image
```

## Skeleton Quick Map

### Body (18 points)
```
Left Arm:   Shoulder(5) → Elbow(7) → Wrist(9)
Right Arm:  Shoulder(6) → Elbow(8) → Wrist(10)
Left Leg:   Hip(11) → Knee(13) → Ankle(15)
Right Leg:  Hip(12) → Knee(14) → Ankle(16)
Head:       Neck(17) → Shoulders(5,6) & Hips(11,12)
Face:       Nose(0) → Eyes(1,2) → Ears(3,4)
```

### Hand (21 points)
```
Palm (0) branches to:
  Thumb (1→2→3→4)
  Index (5→6→7→8)
  Middle (9→10→11→12)
  Ring (13→14→15→16)
  Pinky (17→18→19→20)
```

### Face (68-70 points)
```
No fixed hierarchy
Uses closest neighbor as anchor
```

## Common Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `reduce_confidence` | `True` | Restored points marked as lower confidence |
| `reduce_confidence` | `False` | Restored points inherit full confidence |
| `confidence_reduction_factor` | `1.0` | No reduction (same as parent) |
| `confidence_reduction_factor` | `0.5` | Restored = parent × 0.5 |
| `confidence_reduction_factor` | `0.7` | Default: Restored = parent × 0.7 |

## Key Concepts

### Relative = Proportional
```
Reference:     Current:           Restored:
Shoulder→      Shoulder→          Shoulder→
(+50, -50)     Elbow (scaled)      (+60, -60)
Elbow          Elbow→?            Elbow→
               (scale detected)    (+70, -70)
                                  Wrist
```

### Affine = Transform
```
Detects how skeleton changed:
✓ Rotation (arm twisted)
✓ Scaling (person zoomed in/out)
✓ Translation (person moved)

Applies to offset vectors → maintains proportions
```

### Out-of-Canvas = Precise Internally
```
Keypoint at (600, 300) on 512×512 canvas:
  Internally: Kept as (600, 300) for calculations
  Visualization: Zeroed out for drawing (treated as missing)
  Result: Accurate restoration + bounded rendering
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Keypoints not restoring | Parent missing | Ensure parent exists in current pose |
| Restored points out of order | Hierarchy mismatch | Verify BODY_HIERARCHY matches skeleton |
| Low confidence restored | By design | Set `reduce_confidence=False` if unwanted |
| Wrong offset direction | Affine failed | Check reference pose quality |
| Canvas overflow in render | Canvas too small | Increase canvas size or reduce pose scale |

## Code Examples

### Basic Restoration
```python
restorer = DwRestorator()
image, restored_pose = restorer.dwrestore(
    pose_keypoints=current_pose,
    ref_pose=reference_pose
)
```

### With Confidence Control
```python
image, restored_pose = restorer.dwrestore(
    pose_keypoints=current_pose,
    ref_pose=reference_pose,
    reduce_confidence=True,
    confidence_reduction_factor=0.5  # More aggressive reduction
)
```

### No Confidence Reduction
```python
image, restored_pose = restorer.dwrestore(
    pose_keypoints=current_pose,
    ref_pose=reference_pose,
    reduce_confidence=False
)
```

## Keypoint Format

### Flat List (What you'll see)
```python
[x₀, y₀, c₀, x₁, y₁, c₁, ..., x₁₇, y₁₇, c₁₇]
# For 18 body keypoints: 54 values total
```

### Internal Tuple Format
```python
[(x₀, y₀, c₀), (x₁, y₁, c₁), ..., (x₁₇, y₁₇, c₁₇)]
# Converted internally for easier access
```

### Accessing in JSON
```python
pose_data = {
    "pose_keypoints_2d": [x₀, y₀, c₀, x₁, y₁, c₁, ...],
    "hand_left_keypoints_2d": [21 values for left hand],
    "hand_right_keypoints_2d": [21 values for right hand],
    "face_keypoints_2d": [68-70 values for face]
}
```

## Performance Notes

- **Speed**: ~1-2 ms per pose (18+42+68 keypoints)
- **Accuracy**: Precise to subpixel (float coordinates)
- **Memory**: Creates visualization copy of pose data
- **Scalability**: Linear O(n) in number of keypoints

## Debug Output Example

```
=== DwRestorator: Relative Restoration ===
reduce_confidence: True, factor: 0.7

--- Restoring Body Keypoints ---
  Restored keypoint 9 from parent 7: (460.12, 140.45, 0.630)
  Restored keypoint 7 from parent 5: (380.50, 175.30, 0.700)

--- Restoring Left Hand Keypoints ---
  Restored keypoint 2 from parent 1: (125.43, 89.20, 0.490)

--- Restoring Right Hand Keypoints ---
  [No missing keypoints]

--- Restoring Face Keypoints ---
  Restored face keypoint 45 from anchor 30: (298.76, 234.12, 0.560)

=== Restoration complete ===

=== Generating Pose Image ===
Canvas size: 512x512
Decoded 1 poses from data
Drew poses on canvas
=== Image Generation Complete ===
```

## File Structure

```
dwpose_restorator/
├── nodes.py                          # Main implementation (562 lines)
├── IMPLEMENTATION_GUIDE.md           # High-level overview & concepts
├── TECHNICAL_ARCHITECTURE.md         # Detailed technical design
├── CHANGES.md                        # What was changed
├── demonstration.py                  # Runnable examples
└── QUICK_REFERENCE.md               # This file
```

## Related Files

- OpenPose format: `__init__.py` in controlnet_aux/dwpose/
- Skeleton definitions: See `body.py` limbSeq
- Visualization: Uses DWPose `draw_poses()` function
- Type definitions: `types.py` in controlnet_aux/dwpose/

## Contact & Issues

For issues or questions about the relative restoration:
1. Check TECHNICAL_ARCHITECTURE.md for detailed explanations
2. Run demonstration.py to see concepts in action
3. Review IMPLEMENTATION_GUIDE.md for algorithm details
4. Check debug output in node logs
