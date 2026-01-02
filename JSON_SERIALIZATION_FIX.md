# JSON Serialization Fix for DWPoseRestorator

## Problem
The `SavePoseKpsAsJsonFile` node was failing with the error:
```
TypeError: Object of type float32 is not JSON serializable
```

This occurred because numpy and torch numeric types (e.g., `np.float32`, `np.float64`) cannot be directly serialized to JSON.

## Root Cause
In the `nodes.py` file, the `_restore_keypoints_relative()` and `_restore_face_keypoints()` methods performed arithmetic operations on keypoint coordinates:

```python
x_restored = x_parent_cur + offset_transformed[0]  # Results in numpy/torch type
```

When these values were accumulated and returned, they maintained their numpy types. The `dwrestore()` method then returned this data structure to ComfyUI, which tried to serialize it to JSON using the `SavePoseKpsAsJsonFile` node.

## Solution
Three layers of conversion were implemented to ensure all numeric types are native Python floats:

### 1. **Immediate Type Conversion During Restoration**
In `_restore_keypoints_relative()` and `_restore_face_keypoints()`:
```python
x_restored = float(x_parent_cur + offset_transformed[0])
y_restored = float(y_parent_cur + offset_transformed[1])
c_restored = float(min(c_parent_cur, c_child_ref))
```

### 2. **List Reconstruction with Native Floats**
When converting restored keypoint lists back to flat format:
```python
pose_in_new = []
for x, y, c in restored_body:
    pose_in_new.extend([float(x), float(y), float(c)])  # Explicit conversion
pin["pose_keypoints_2d"] = pose_in_new
```

This is done for:
- Body keypoints
- Left hand keypoints
- Right hand keypoints
- Face keypoints

### 3. **Final Deep Conversion with Utility Function**
After all restoration, a recursive conversion function is applied:

```python
def convert_to_python_types(obj):
    """Recursively convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        obj = obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    return obj
```

This is called before returning the restored pose data:
```python
out_for_export = convert_to_python_types(out_for_export)
```

## Files Modified
- `nodes.py`: Added type conversion throughout the restoration process

## Testing
The fix ensures that:
1. All keypoint coordinates are native Python floats (not numpy types)
2. All nested dictionaries and lists are recursively converted
3. The output is fully JSON-serializable
4. The `SavePoseKpsAsJsonFile` node can successfully serialize the pose data

## Backward Compatibility
This change is fully backward compatible:
- Existing pose data structures are preserved
- Only the internal Python type representation is changed
- Output format remains identical
- No API changes to the node inputs/outputs
