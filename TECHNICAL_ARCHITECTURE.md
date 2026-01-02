# Technical Architecture: Relative Restoration System

## System Design Overview

```
Input: Current Pose (partial)  +  Reference Pose (complete/better)
                    ↓
        ┌───────────────────────┐
        │  Extract Keypoints    │
        │  (flat → tuples)      │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────────────┐
        │ Estimate Affine Transform     │
        │ (Find rotation+scale+trans)   │
        └───────────┬───────────────────┘
                    ↓
        ┌───────────────────────────────┐
        │ For Each Missing Keypoint:    │
        │ 1. Find parent (hierarchy)    │
        │ 2. Get reference offset       │
        │ 3. Transform offset (affine)  │
        │ 4. Apply to parent position   │
        │ 5. Set confidence             │
        └───────────┬───────────────────┘
                    ↓
        ┌───────────────────────────────┐
        │  Visualization Clamping       │
        │  (canvas bounds only)         │
        └───────────┬───────────────────┘
                    ↓
Output: Restored Pose (complete)  +  Visualization Image
```

## Core Components

### 1. Keypoint Missing Detection

```python
def _is_keypoint_missing(x, y, confidence):
    """Detects missing keypoints by all-zero condition"""
    return x == 0.0 and y == 0.0 and confidence == 0.0
```

**Design Decision**: Using all-zero check matches DWPose output format where missing keypoints are represented as (0, 0, 0).

### 2. Affine Transformation Estimation

```
Input:  Keypoints (current & reference)
        ↓
    Find pairs where:
    - Reference has high confidence (c > 0.3)
    - Current has high confidence (c > 0.3)
        ↓
    Collect: src_points (reference), dst_points (current)
        ↓
    Need minimum 3 pairs for affine
        ↓
    Option A: cv2.getAffineTransform(src[:3], dst[:3])
    Option B: np.linalg.lstsq for >3 points
        ↓
    Return: 2x3 transformation matrix or None
```

**Why OpenCV?**
- `cv2.getAffineTransform`: Direct 3-point solution, always works
- `np.linalg.lstsq`: Better for >3 points, reduces outlier influence

**Matrix Form:**
```
[a  b  tx]   [x]   [x']
[c  d  ty] × [y] = [y']
             [1]
```

### 3. Offset Transformation

```
Reference:
  parent_ref = (px_ref, py_ref)
  child_ref = (cx_ref, cy_ref)
  offset_ref = (cx_ref - px_ref, cy_ref - py_ref)

Current:
  parent_cur = (px_cur, py_cur)
  [Affine transform applied to offset]
  offset_cur = A × offset_ref
  child_cur = parent_cur + offset_cur
```

**Critical Detail**: When transforming an offset, we apply the full affine (rotation + scale) but NOT the translation component, because the offset is relative (relative to parent position).

```python
# Full transform
transformed_full = affine @ [offset_x, offset_y, 1]

# Remove translation component for offset
offset_transformed = (
    transformed_full[0] - affine[0, 2],
    transformed_full[1] - affine[1, 2]
)
```

### 4. Hierarchical Processing

**Processing Order** (within each hierarchy):
- Restores in dependency order: parents before children
- For example, shoulder → elbow → wrist (left to right)

**Cascading Restoration**:
```
Missing: Elbow, Wrist
Step 1: Restore Elbow from Shoulder
Step 2: Restore Wrist from Elbow (using newly restored position)
```

This is automatic because we iterate in hierarchy order.

### 5. Confidence Handling

```
Confidence Source Priority:
  1. Parent's confidence (current pose)
  2. OR Child's confidence in reference (if parent not available)
  3. Optional reduction: confidence *= reduction_factor

Example:
  Parent confidence: 0.9
  Restored child confidence: 0.9
  With reduction (0.7x): 0.63
```

**Why Inheritance?**
- Restored keypoints are less reliable than detected ones
- Using parent's confidence indicates "as reliable as what connected to it"
- Reduction factor allows marking as "lower confidence restoration"

### 6. Visualization / Export Policy

```python
def _zero_out_of_canvas(pose_data, canvas_height, canvas_width):
  """
  Zero-out out-of-canvas keypoints in a copy used for visualization/export.
  Preserves internal values for calculations.
  """
  for each_keypoint:
    if x < 0 or x >= width or y < 0 or y >= height:
      x, y, conf = 0.0, 0.0, 0.0
```

**Key Design Decision**: Create a COPY of pose data for visualization and export, zero-out any keypoints that fall outside the canvas bounds, and keep original values internally. This preserves:
- Out-of-canvas keypoint coordinates internally for accurate restorations
- Clean visualization and exported JSON (missing markers instead of invalid coordinates)
- Accurate downstream processing that expects either in-canvas coordinates or explicit missing points

### 7. GPU Fallback

The implementation supports optional GPU usage for tensor creation and processing. Behavior:
- If `use_gpu` is requested and `torch.cuda.is_available()` is True, tensors are created on CUDA device.
- Otherwise, CPU is used. This ensures the node works on machines without GPU.

## Skeleton Hierarchies

### Body (18 keypoints)

```
Index  Name              Parent
─────────────────────────────────
  0    Nose              (root)
  1    Left Eye          0 (Nose)
  2    Right Eye         0 (Nose)
  3    Left Ear          1 (Left Eye)
  4    Right Ear         2 (Right Eye)
  5    Left Shoulder     17 (Neck)
  6    Right Shoulder    17 (Neck)
  7    Left Elbow        5 (Left Shoulder)
  8    Right Elbow       6 (Right Shoulder)
  9    Left Wrist        7 (Left Elbow)
 10    Right Wrist       8 (Right Elbow)
 11    Left Hip          17 (Neck)
 12    Right Hip         17 (Neck)
 13    Left Knee         11 (Left Hip)
 14    Right Knee        12 (Right Hip)
 15    Left Ankle        13 (Left Knee)
 16    Right Ankle       14 (Right Knee)
 17    Neck              (root)
```

**Restoration Order** (as used in `BODY_HIERARCHY.items()`, which iterates depth-first):
1. Eyes and Ears (from Nose/Eyes)
2. Shoulders (from Neck)
3. Elbows (from Shoulders)
4. Wrists (from Elbows)
5. Hips (from Neck)
6. Knees (from Hips)
7. Ankles (from Knees)

### Hand (21 keypoints each)

```
Palm Center (0)
│
├─ Thumb Chain:     1→2→3→4
├─ Index Chain:     5→6→7→8
├─ Middle Chain:    9→10→11→12
├─ Ring Chain:      13→14→15→16
└─ Pinky Chain:     17→18→19→20
```

**Property**: All fingers branch from palm center, enabling parallel restoration.

### Face (68-70 keypoints)

```
No fixed hierarchy. Instead:
- Find face center (center of existing keypoints)
- For each missing keypoint:
  - Find closest existing keypoint as anchor
  - Restore relative to that anchor
```

**Advantage**: Works with arbitrary face poses, doesn't assume fixed structure.

## Error Handling & Edge Cases

### Case 1: Insufficient Existing Keypoints
```
If < 3 existing keypoints → Can't estimate affine
Solution: affine_matrix = None
Effect: Use identity transform (offset_transformed = offset_ref)
```

### Case 2: Parent Doesn't Exist in Current Pose
```
If parent missing in current → Can't apply offset
Solution: Skip child restoration
Effect: Child remains missing
```

### Case 3: Missing Keypoint Has Partial Data
```
If x=0, y=0, c≠0 (or similar) → Treated as existing
Solution: _is_keypoint_missing requires ALL THREE to be 0
Effect: Won't overwrite partially valid data
```

### Case 4: Reference is Completely Empty
```
If ref_pose is None → Early return
Effect: Return unchanged pose_keypoints
```

### Case 5: Out-of-Canvas After Transformation
```
If x_restored < 0 or > canvas_width → Keep internal value
Solution: Clamp only in visualization copy
Effect: Full precision maintained for calculations, visualization bounded
```

## Performance Characteristics

### Time Complexity
- Affine estimation: O(n) where n = number of keypoints
- Restoration loop: O(m) where m = number of missing keypoints
- Overall: **O(n)** linear in number of keypoints

### Space Complexity
- Affine matrix: O(1) constant (2x3)
- Keypoint lists: O(n) for duplicated data
- Visualization copy: O(n) additional
- Overall: **O(n)** linear

### Computation Count
- Per frame (~18 body + 42 hand + 68 face = 128 keypoints):
  - Affine: ~3 matrix operations per hierarchy
  - Restoration: ~1-2 array operations per missing point
  - Typical: <1ms on modern CPU

## Future Optimization Opportunities

1. **Batch Processing**: Process multiple people at once
2. **GPU Acceleration**: Use torch for matrix operations
3. **Caching**: Cache affine transforms if same pose detected
4. **Adaptive Hierarchies**: Learn hierarchies from data
5. **Confidence Weighting**: Weight existing keypoints by confidence in affine estimation
6. **Temporal Consistency**: Use temporal information across frames

## Testing Strategy

### Unit Tests
- Affine transformation with known matrices
- Offset transformation correctness
- Hierarchy correctness for each body part
- Out-of-canvas boundary conditions
- Confidence inheritance logic

### Integration Tests
- Full restoration pipeline with mock poses
- Canvas clamping verification
- Visualization generation
- Edge cases (empty reference, all missing, etc.)

### Validation
- Visual inspection of restored poses
- Proportion preservation verification
- Comparison with baseline absolute restoration
- Performance benchmarking
