# DWPose Relative Restoration Implementation Guide

## Overview

The `DwRestorator` node has been completely redesigned to use **relative, hierarchical restoration** based on skeleton structure and affine transformations. This approach maintains body proportions and keypoint relationships even when the pose is at different positions, scales, or rotations compared to the reference.

## Key Concepts

### 1. **Relative Restoration**
Instead of copying absolute coordinates from a reference, the system now:
- Identifies existing keypoints in the current pose
- For missing keypoints, uses parent-child relationships from the skeleton
- Calculates the offset vector from parent to child in the reference
- **Transforms** this offset using an affine transformation to match the current pose's scale/rotation
- Applies the transformed offset to the current parent's position

### 2. **Affine Transformation Estimation**
The system automatically detects how the pose has changed relative to the reference:
- Compares positions of existing keypoints between current and reference
- Uses least-squares fitting with OpenCV's affine transformation
- Captures: **rotation**, **scaling**, and **translation**
- This transformation is then applied to missing keypoint offsets

### 3. **Skeleton Hierarchies**

#### Body Keypoints (18 points - OpenPose COCO format)
```
Nose (0)
├── Left Eye (1) → Left Ear (3)
└── Right Eye (2) → Right Ear (4)

Neck (17)
├── Left Shoulder (5) → Left Elbow (7) → Left Wrist (9)
├── Right Shoulder (6) → Right Elbow (8) → Right Wrist (10)
├── Left Hip (11) → Left Knee (13) → Left Ankle (15)
└── Right Hip (12) → Right Knee (14) → Right Ankle (16)
```

#### Hand Keypoints (21 points each hand)
```
Palm Center (0)
├── Thumb: (1) → (2) → (3) → (4)
├── Index: (5) → (6) → (7) → (8)
├── Middle: (9) → (10) → (11) → (12)
├── Ring: (13) → (14) → (15) → (16)
└── Pinky: (17) → (18) → (19) → (20)
```

#### Face Keypoints (68-70 points)
- Uses **local hierarchy**: finds closest existing keypoint as anchor
- Restores missing face points relative to their nearest neighbor

## Algorithm Flow

### For Each Keypoint Type (Body, Hands, Face):

1. **Extract Keypoints**
   - Convert flat list `[x, y, conf, x, y, conf, ...]` to `[(x, y, conf), ...]` tuples

2. **Estimate Affine Transform**
   - Find keypoints that exist in both current and reference poses
   - Use these pairs to estimate best-fit affine transformation
   - Requires minimum 3 existing keypoint pairs
   - Uses least-squares fitting for accuracy

3. **Restore Missing Keypoints**
   - For each missing keypoint (detected by `x==0, y==0, conf==0`):
     - **Skip if** it already exists in current pose
     - **Find parent** from hierarchy definition
     - **Verify parent exists** in current pose (if not, skip)
     - **Get offset** from parent→child in reference
     - **Transform offset** using affine matrix (rotation + scale)
     - **Apply to current parent** position
     - **Set confidence** to parent's confidence (or lower if requested)

4. **Visualization / Export Policy**
   - Create a copy of pose data for visualization and exported output
   - Zero-out any keypoints that fall outside the canvas bounds `[0, width)` and `[0, height)` in that copy
   - Keep original unclamped/out-of-canvas values internally for calculations (affine estimation, chain restorations)
   - This preserves geometric accuracy while ensuring visualization and exported JSON contain only in-canvas or explicit-missing markers

## New Features

### Parameters
- `reduce_confidence` (bool, default=True): Lower confidence of restored keypoints
- `confidence_reduction_factor` (float, 0.0-1.0, default=0.7): Multiplier for confidence

### Key Methods

#### `_is_keypoint_missing(x, y, confidence)`
Returns `True` if keypoint is missing (all zeros)

#### `_estimate_affine_transform(keypoints_current, keypoints_ref)`
Uses OpenCV and numpy least-squares to find best-fit transformation

#### `_transform_point(point, affine_matrix)`
Applies affine transformation to a 2D point

#### `_restore_keypoints_relative(keypoints_current, keypoints_ref, hierarchy, ...)`
Main restoration logic for body/hands using parent-child relationships

#### `_restore_face_keypoints(keypoints_current, keypoints_ref, ...)`
Specialized restoration for face keypoints using local hierarchy

#### `_zero_out_of_canvas(pose_data, height, width)`
Zero-out out-of-canvas keypoints in a pose copy so visualization and exported JSON treat them as missing

## Example Scenario

**Reference Pose**: Person standing upright with right hand raised
- Right shoulder at (300, 200)
- Right elbow at (350, 150) - 50 pixels to the right and 50 up
- Right wrist is MISSING (0, 0, 0)

**Current Pose**: Same person, but shifted right and slightly scaled
- Right shoulder at (400, 200)
- Right elbow is MISSING
- Right wrist is MISSING

**Process**:
1. Estimate affine: shoulder (300→400 in current) - calculate transformation
2. Restore elbow:
   - Parent: shoulder at (400, 200) in current
   - Reference offset: (50, -50)
   - Apply affine to offset (e.g., scale by 1.2): (60, -60)
   - Result: (400+60, 200-60) = **(460, 140)**
3. Restore wrist:
   - Parent: elbow at (460, 140) in current
   - Reference offset: (50, -100)
   - Apply affine to offset: (60, -120)
   - Result: **(520, 20)** ← Out of canvas but used internally

## Technical Details

### Why Affine Transform?
- Handles rotation, scaling, and translation simultaneously
- Captures how the pose has changed relative to reference
- Applied to offset vectors maintains proportions
- Works even if person moved, rotated, or changed apparent size

### Why Confidence Inheritance?
- Restored keypoints inherit confidence from their parent
- Optional reduction factor acknowledges lower reliability of restored points
- Allows downstream processes to weight restored vs. detected keypoints

### Why Canvas Clamping?
- Keypoints can legitimately extend beyond canvas during calculation
- Visualization must keep keypoints within drawable bounds
- Unclamped values preserved internally for chain-of-restoration (e.g., wrist depends on elbow)

## Debugging

The implementation includes extensive logging:
```
=== DwRestorator: Relative Restoration ===
--- Restoring Body Keypoints ---
  Restored keypoint 9 from parent 7: (460.00, 140.00, 0.700)
...
=== Restoration complete ===
=== Generating Pose Image ===
Canvas size: 512x512
...
```

## Edge Cases Handled

1. **Reference has no data**: Falls back to current pose
2. **Insufficient existing keypoints**: Skips affine estimation, uses identity transform
3. **Parent doesn't exist in current pose**: Skips child restoration
4. **Chain restoration**: Wrist restoration depends on elbow, elbow on shoulder
5. **Out-of-canvas keypoints**: Zeroed in visualization/export copy, preserved internally for calculations

## Performance Considerations

- Affine transform estimated once per keypoint type (~3-4 times per pose)
- O(n) restoration where n = number of missing keypoints
- Minimal overhead vs. original approach
- All operations use numpy vectorization where possible
