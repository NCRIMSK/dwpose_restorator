import copy
import numpy as np
import torch
import sys
import os
import cv2
import json

# Try to import DWPose utilities - local first, fallback to controlnet_aux
try:
    # Try relative import (when loaded as package)
    from .pose_visualization import (
        decode_json_as_poses,
        draw_poses
    )
    DWPOSE_AVAILABLE = True
    print("[DWRestorator] Using local pose visualization module (relative import)")
except ImportError as e1:
    try:
        # Try absolute import (when loaded as standalone)
        from pose_visualization import (
            decode_json_as_poses,
            draw_poses
        )
        DWPOSE_AVAILABLE = True
        print("[DWRestorator] Using local pose visualization module (absolute import)")
    except ImportError as e2:
        try:
            from custom_controlnet_aux.dwpose import ( # pyright: ignore[reportMissingImports]
                decode_json_as_poses,
                draw_poses
            )
            DWPOSE_AVAILABLE = True
            print("[DWRestorator] Using controlnet_aux pose visualization module")
        except ImportError as e3:
            print(f"[DWRestorator] WARNING: Pose visualization not available")
            print(f"[DWRestorator]   - Relative import failed: {e1}")
            print(f"[DWRestorator]   - Absolute import failed: {e2}")
            print(f"[DWRestorator]   - Controlnet_aux import failed: {e3}")
            print("[DWRestorator] Will use fallback blank image generation")
            DWPOSE_AVAILABLE = False

# DWPose keypoint hierarchies (0-indexed)
# OpenPose COCO format: 18 body keypoints
BODY_HIERARCHY = {
    # Format: child_idx: parent_idx
    1: 0,   # left_eye -> nose
    2: 0,   # right_eye -> nose
    3: 1,   # left_ear -> left_eye
    4: 2,   # right_ear -> right_eye
    5: 17,  # left_shoulder -> neck
    6: 17,  # right_shoulder -> neck
    7: 5,   # left_elbow -> left_shoulder
    8: 6,   # right_elbow -> right_shoulder
    9: 7,   # left_wrist -> left_elbow
    10: 8,  # right_wrist -> right_elbow
    11: 17, # left_hip -> neck
    12: 17, # right_hip -> neck
    13: 11, # left_knee -> left_hip
    14: 12, # right_knee -> right_hip
    15: 13, # left_ankle -> left_knee
    16: 14, # right_ankle -> right_knee
}

# Hand hierarchy (0-indexed, 21 keypoints)
# Palm center (0) -> finger bases (1-4) -> finger tips (5-20)
HAND_HIERARCHY = {
    1: 0, 2: 1, 3: 2, 4: 3,    # Thumb
    5: 0, 6: 5, 7: 6, 8: 7,    # Index
    9: 0, 10: 9, 11: 10, 12: 11,  # Middle
    13: 0, 14: 13, 15: 14, 16: 15,  # Ring
    17: 0, 18: 17, 19: 18, 20: 19,  # Pinky
}

# Face hierarchy - use local regions with face center as anchor
# Simplified: group landmarks by facial regions
FACE_CENTER_IDX = 33  # Nose tip as center (approximate)


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


class DwRestorator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pose_keypoints": ("POSE_KEYPOINT",)},
            "optional": {
                "ref_pose": ("POSE_KEYPOINT",),
                "reduce_confidence": ("BOOLEAN", {"default": True}),
                "confidence_reduction_factor": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_image", "pose_restored",)
    CATEGORY = "DWPoseRestorator"
    FUNCTION = "dwrestore"

    def _is_keypoint_missing(self, x, y, confidence):
        """Check if a keypoint is missing (zero coordinates and low/zero confidence)"""
        return x == 0.0 and y == 0.0 and confidence == 0.0

    def _estimate_affine_transform(self, keypoints_current, keypoints_ref):
        """
        Estimate affine transformation (rotation + scale + translation) from existing keypoints.
        Uses least squares fitting on existing keypoints to find the best transformation.
        
        Args:
            keypoints_current: List of (x, y, conf) for current pose
            keypoints_ref: List of (x, y, conf) for reference pose
            
        Returns:
            affine_matrix: 2x3 affine transformation matrix, or None if insufficient points
        """
        # Find keypoints that exist in both current and reference
        src_points = []
        dst_points = []
        
        for i in range(min(len(keypoints_current), len(keypoints_ref))):
            x_cur, y_cur, c_cur = keypoints_current[i]
            x_ref, y_ref, c_ref = keypoints_ref[i]
            
            # Use keypoints that exist in reference (high confidence)
            if c_ref > 0.3 and not self._is_keypoint_missing(x_ref, y_ref, c_ref):
                # If current keypoint exists, use it as target
                if c_cur > 0.3 and not self._is_keypoint_missing(x_cur, y_cur, c_cur):
                    src_points.append([x_ref, y_ref])
                    dst_points.append([x_cur, y_cur])
        
        if len(src_points) < 3:
            # Not enough points to estimate transformation
            return None
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Estimate affine transformation
        affine_matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3]) if len(src_points) >= 3 else None
        
        # For better results with more points, use least squares
        if len(src_points) > 3:
            try:
                # Fit affine using more points
                A = np.vstack([src_points.T, np.ones(len(src_points))])
                b_x = dst_points[:, 0]
                b_y = dst_points[:, 1]
                
                # Solve for transformation matrix
                # [a b tx] * [x]   [x']
                # [c d ty]   [y] = [y']
                #            [1]
                coeff_x = np.linalg.lstsq(A.T, b_x, rcond=None)[0]
                coeff_y = np.linalg.lstsq(A.T, b_y, rcond=None)[0]
                
                affine_matrix = np.array([coeff_x, coeff_y], dtype=np.float32)
            except:
                pass
        
        return affine_matrix

    def _transform_point(self, point, affine_matrix):
        """Apply affine transformation to a point"""
        if affine_matrix is None:
            return point
        
        x, y = point
        # affine_matrix is 2x3: [[a, b, tx], [c, d, ty]]
        x_new = affine_matrix[0, 0] * x + affine_matrix[0, 1] * y + affine_matrix[0, 2]
        y_new = affine_matrix[1, 0] * x + affine_matrix[1, 1] * y + affine_matrix[1, 2]
        return (x_new, y_new)

    def _restore_keypoints_relative(self, keypoints_current, keypoints_ref, hierarchy, 
                                   reduce_confidence=True, confidence_factor=0.7):
        """
        Restore missing keypoints using relative hierarchy and affine transformation.
        
        Args:
            keypoints_current: List of (x, y, conf) for current pose
            keypoints_ref: List of (x, y, conf) for reference pose
            hierarchy: Dict mapping child_idx -> parent_idx
            reduce_confidence: Whether to reduce confidence of restored keypoints
            confidence_factor: Factor to multiply confidence (0.0-1.0)
            
        Returns:
            Restored keypoints list
        """
        if keypoints_ref is None:
            return keypoints_current
        
        restored = copy.deepcopy(keypoints_current)
        
        # Estimate affine transformation
        affine_matrix = self._estimate_affine_transform(keypoints_current, keypoints_ref)
        
        # Process each keypoint
        for child_idx, parent_idx in hierarchy.items():
            if child_idx >= len(restored):
                continue
            
            x_child, y_child, c_child = restored[child_idx]
            
            # Skip if keypoint already exists
            if not self._is_keypoint_missing(x_child, y_child, c_child):
                continue
            
            # Try to restore from parent
            if parent_idx < len(keypoints_current) and parent_idx < len(keypoints_ref):
                x_parent_cur, y_parent_cur, c_parent_cur = keypoints_current[parent_idx]
                x_parent_ref, y_parent_ref, c_parent_ref = keypoints_ref[parent_idx]
                
                # Parent must exist in current pose
                if self._is_keypoint_missing(x_parent_cur, y_parent_cur, c_parent_cur):
                    continue
                
                # Get offset in reference
                x_child_ref, y_child_ref, c_child_ref = keypoints_ref[child_idx]
                
                if not self._is_keypoint_missing(x_child_ref, y_child_ref, c_child_ref):
                    # Calculate offset vector in reference
                    offset_x = x_child_ref - x_parent_ref
                    offset_y = y_child_ref - y_parent_ref
                    
                    # Transform offset using affine matrix
                    if affine_matrix is not None:
                        # Apply rotation and scaling to offset
                        offset_transformed = self._transform_point((offset_x, offset_y), affine_matrix)
                        # Remove translation component
                        offset_transformed = (
                            offset_transformed[0] - affine_matrix[0, 2],
                            offset_transformed[1] - affine_matrix[1, 2]
                        )
                    else:
                        offset_transformed = (offset_x, offset_y)
                    
                    # Apply to current parent position
                    x_restored = float(x_parent_cur + offset_transformed[0])
                    y_restored = float(y_parent_cur + offset_transformed[1])
                    
                    # Use parent's confidence or reference's confidence, then optionally reduce
                    c_restored = float(min(c_parent_cur, c_child_ref))
                    if reduce_confidence:
                        c_restored = float(c_restored * confidence_factor)
                    
                    restored[child_idx] = [x_restored, y_restored, c_restored]
                    print(f"  Restored keypoint {child_idx} from parent {parent_idx}: ({x_restored:.2f}, {y_restored:.2f}, {c_restored:.3f})")
        
        return restored

    def dwrestore(self, pose_keypoints, ref_pose=None, reduce_confidence=True, confidence_reduction_factor=0.7, use_gpu=False):
        print(f"\n=== DwRestorator: Relative Restoration ===")
        print(f"pose_keypoints type: {type(pose_keypoints)}")
        print(f"ref_pose type: {type(ref_pose)}")
        print(f"reduce_confidence: {reduce_confidence}, factor: {confidence_reduction_factor}")
        
        if ref_pose is None:
            print("ERROR: ref_pose is None, returning unchanged")
            return (self._create_blank_image(), pose_keypoints)

        out = copy.deepcopy(pose_keypoints)
        ref = ref_pose

        def get_person0(x):
            # x may be a top-level dict containing "people": [...]
            if isinstance(x, dict) and "people" in x and isinstance(x["people"], list) and x["people"]:
                return x["people"][0], "dict_people0"

            # x may be a list. It can be either:
            # - a list of person dicts (people list) -> [ {pose..}, ... ]
            # - a list containing a top-level dict like {"people": [ {...} ], ...}
            if isinstance(x, list) and x:
                first = x[0]
                # list contains a top-level dict with "people"
                if isinstance(first, dict) and "people" in first and isinstance(first["people"], list) and first["people"]:
                    return first["people"][0], "list_topdict_people0"
                # list directly contains person dict(s)
                if isinstance(first, dict) and ("pose_keypoints_2d" in first or "face_keypoints_2d" in first or "hand_left_keypoints_2d" in first or "hand_right_keypoints_2d" in first):
                    return first, "list_people0"

            raise TypeError(f"Unsupported POSE_KEYPOINT structure: {type(x)}")

        try:
            pin, mode_in = get_person0(out)
            pref, mode_ref = get_person0(ref)
            print(f"Extracted person dicts successfully")
            print(f"Input person keys: {list(pin.keys())}")
            print(f"Reference person keys: {list(pref.keys())}")
        except Exception as e:
            print(f"ERROR extracting person data: {e}")
            return (self._create_blank_image(), out)

        # Restore body keypoints
        print(f"\n--- Restoring Body Keypoints ---")
        pose_in = pin.get("pose_keypoints_2d", None)
        pose_ref = pref.get("pose_keypoints_2d", None)
        
        if pose_in and isinstance(pose_in, list) and len(pose_in) >= 3:
            # Convert flat list to (x, y, conf) tuples
            keypoints_current = [(pose_in[i], pose_in[i+1], pose_in[i+2]) for i in range(0, len(pose_in)-2, 3)]
            keypoints_ref_list = None
            if pose_ref and isinstance(pose_ref, list) and len(pose_ref) >= 3:
                keypoints_ref_list = [(pose_ref[i], pose_ref[i+1], pose_ref[i+2]) for i in range(0, len(pose_ref)-2, 3)]
            
            restored_body = self._restore_keypoints_relative(
                keypoints_current, keypoints_ref_list, BODY_HIERARCHY,
                reduce_confidence, confidence_reduction_factor
            )
            
            # Convert back to flat list with native Python floats
            pose_in_new = []
            for x, y, c in restored_body:
                pose_in_new.extend([float(x), float(y), float(c)])
            pin["pose_keypoints_2d"] = pose_in_new

        # Restore left hand keypoints
        print(f"\n--- Restoring Left Hand Keypoints ---")
        hand_left_in = pin.get("hand_left_keypoints_2d", None)
        hand_left_ref = pref.get("hand_left_keypoints_2d", None)
        
        if hand_left_in and isinstance(hand_left_in, list) and len(hand_left_in) >= 3:
            keypoints_current = [(hand_left_in[i], hand_left_in[i+1], hand_left_in[i+2]) for i in range(0, len(hand_left_in)-2, 3)]
            keypoints_ref_list = None
            if hand_left_ref and isinstance(hand_left_ref, list) and len(hand_left_ref) >= 3:
                keypoints_ref_list = [(hand_left_ref[i], hand_left_ref[i+1], hand_left_ref[i+2]) for i in range(0, len(hand_left_ref)-2, 3)]
            
            restored_hand = self._restore_keypoints_relative(
                keypoints_current, keypoints_ref_list, HAND_HIERARCHY,
                reduce_confidence, confidence_reduction_factor
            )
            
            hand_left_in_new = []
            for x, y, c in restored_hand:
                hand_left_in_new.extend([float(x), float(y), float(c)])
            pin["hand_left_keypoints_2d"] = hand_left_in_new

        # Restore right hand keypoints
        print(f"\n--- Restoring Right Hand Keypoints ---")
        hand_right_in = pin.get("hand_right_keypoints_2d", None)
        hand_right_ref = pref.get("hand_right_keypoints_2d", None)
        
        if hand_right_in and isinstance(hand_right_in, list) and len(hand_right_in) >= 3:
            keypoints_current = [(hand_right_in[i], hand_right_in[i+1], hand_right_in[i+2]) for i in range(0, len(hand_right_in)-2, 3)]
            keypoints_ref_list = None
            if hand_right_ref and isinstance(hand_right_ref, list) and len(hand_right_ref) >= 3:
                keypoints_ref_list = [(hand_right_ref[i], hand_right_ref[i+1], hand_right_ref[i+2]) for i in range(0, len(hand_right_ref)-2, 3)]
            
            restored_hand = self._restore_keypoints_relative(
                keypoints_current, keypoints_ref_list, HAND_HIERARCHY,
                reduce_confidence, confidence_reduction_factor
            )
            
            hand_right_in_new = []
            for x, y, c in restored_hand:
                hand_right_in_new.extend([float(x), float(y), float(c)])
            pin["hand_right_keypoints_2d"] = hand_right_in_new

        # Restore face keypoints (simplified local hierarchy)
        print(f"\n--- Restoring Face Keypoints ---")
        face_in = pin.get("face_keypoints_2d", None)
        face_ref = pref.get("face_keypoints_2d", None)
        
        if face_in and isinstance(face_in, list) and len(face_in) >= 3:
            keypoints_current = [(face_in[i], face_in[i+1], face_in[i+2]) for i in range(0, len(face_in)-2, 3)]
            keypoints_ref_list = None
            if face_ref and isinstance(face_ref, list) and len(face_ref) >= 3:
                keypoints_ref_list = [(face_ref[i], face_ref[i+1], face_ref[i+2]) for i in range(0, len(face_ref)-2, 3)]
            
            restored_face = self._restore_face_keypoints(
                keypoints_current, keypoints_ref_list,
                reduce_confidence, confidence_reduction_factor
            )
            
            face_in_new = []
            for x, y, c in restored_face:
                face_in_new.extend([float(x), float(y), float(c)])
            pin["face_keypoints_2d"] = face_in_new

        print(f"=== Restoration complete ===\n")
        
        # After restoration, prepare exported pose where out-of-canvas keypoints are zeroed
        out_for_export = copy.deepcopy(out)
        canvas_h, canvas_w = self._get_canvas_dims(out_for_export)
        self._zero_out_of_canvas(out_for_export, canvas_h, canvas_w)
        
        # Convert all numeric types to native Python types for JSON serialization
        out_for_export = convert_to_python_types(out_for_export)

        # Generate image output (visualization uses zeroed copy internally)
        image_output = self._generate_pose_image(out, use_gpu=use_gpu)

        return (image_output, out_for_export,)

    def _restore_face_keypoints(self, keypoints_current, keypoints_ref, 
                               reduce_confidence=True, confidence_factor=0.7):
        """
        Restore face keypoints using simple local hierarchy approach.
        Uses face center (nose) as anchor for nearby regions.
        """
        if keypoints_ref is None:
            return keypoints_current
        
        restored = copy.deepcopy(keypoints_current)
        
        # Estimate affine transformation from existing face keypoints
        affine_matrix = self._estimate_affine_transform(keypoints_current, keypoints_ref)
        
        # Find face center (approximate center of existing keypoints)
        existing_points = []
        for i, (x, y, c) in enumerate(keypoints_current):
            if c > 0.3 and not self._is_keypoint_missing(x, y, c):
                existing_points.append((x, y))
        
        if existing_points:
            face_center = (
                np.mean([p[0] for p in existing_points]),
                np.mean([p[1] for p in existing_points])
            )
        else:
            face_center = None
        
        # Restore missing face keypoints
        for i in range(len(restored)):
            x, y, c = restored[i]
            
            if self._is_keypoint_missing(x, y, c):
                # Try to restore from reference
                if i < len(keypoints_ref):
                    x_ref, y_ref, c_ref = keypoints_ref[i]
                    if not self._is_keypoint_missing(x_ref, y_ref, c_ref):
                        # Find closest existing keypoint to use as anchor
                        closest_idx = None
                        min_dist = float('inf')
                        
                        for j in range(len(keypoints_current)):
                            x_j, y_j, c_j = keypoints_current[j]
                            if c_j > 0.3 and not self._is_keypoint_missing(x_j, y_j, c_j):
                                dist = (x_ref - x_j) ** 2 + (y_ref - y_j) ** 2
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_idx = j
                        
                        if closest_idx is not None:
                            x_anchor, y_anchor, c_anchor = keypoints_current[closest_idx]
                            x_anchor_ref, y_anchor_ref, c_anchor_ref = keypoints_ref[closest_idx]
                            
                            # Calculate offset in reference
                            offset_x = x_ref - x_anchor_ref
                            offset_y = y_ref - y_anchor_ref
                            
                            # Transform offset
                            if affine_matrix is not None:
                                offset_transformed = self._transform_point((offset_x, offset_y), affine_matrix)
                                offset_transformed = (
                                    offset_transformed[0] - affine_matrix[0, 2],
                                    offset_transformed[1] - affine_matrix[1, 2]
                                )
                            else:
                                offset_transformed = (offset_x, offset_y)
                            
                            x_restored = float(x_anchor + offset_transformed[0])
                            y_restored = float(y_anchor + offset_transformed[1])
                            c_restored = float(c_ref)
                            if reduce_confidence:
                                c_restored = float(c_restored * confidence_factor)
                            
                            restored[i] = [x_restored, y_restored, c_restored]
                            print(f"  Restored face keypoint {i} from anchor {closest_idx}: ({x_restored:.2f}, {y_restored:.2f}, {c_restored:.3f})")
        
        return restored


    def _zero_out_of_canvas(self, pose_data, canvas_height, canvas_width):
        """
        Zero-out keypoints that are outside the canvas bounds for visualization and exported output.
        This preserves internal coordinates for calculations (use on a copy when needed).
        """
        try:
            def get_person0(x):
                if isinstance(x, dict) and "people" in x and isinstance(x["people"], list) and x["people"]:
                    return x["people"][0]
                if isinstance(x, list) and x:
                    first = x[0]
                    if isinstance(first, dict) and "people" in first and isinstance(first["people"], list) and first["people"]:
                        return first["people"][0]
                    if isinstance(first, dict) and ("pose_keypoints_2d" in first or "face_keypoints_2d" in first):
                        return first
                return None

            person = get_person0(pose_data)
            if person is None:
                return

            # Body keypoints
            if "pose_keypoints_2d" in person and isinstance(person["pose_keypoints_2d"], list):
                kpts = person["pose_keypoints_2d"]
                for i in range(0, len(kpts) - 2, 3):
                    x, y = kpts[i], kpts[i+1]
                    if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
                        kpts[i], kpts[i+1], kpts[i+2] = 0.0, 0.0, 0.0

            # Hand keypoints
            for hand_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                if hand_key in person and isinstance(person[hand_key], list):
                    kpts = person[hand_key]
                    for i in range(0, len(kpts) - 2, 3):
                        x, y = kpts[i], kpts[i+1]
                        if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
                            kpts[i], kpts[i+1], kpts[i+2] = 0.0, 0.0, 0.0

            # Face keypoints
            if "face_keypoints_2d" in person and isinstance(person["face_keypoints_2d"], list):
                kpts = person["face_keypoints_2d"]
                for i in range(0, len(kpts) - 2, 3):
                    x, y = kpts[i], kpts[i+1]
                    if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
                        kpts[i], kpts[i+1], kpts[i+2] = 0.0, 0.0, 0.0
        except Exception as e:
            print(f"WARNING: Error zeroing out-of-canvas keypoints: {e}")

    def _get_canvas_dims(self, pose_data):
        """Try to extract canvas_height and canvas_width from the pose JSON structure."""
        if isinstance(pose_data, list) and len(pose_data) > 0:
            first_item = pose_data[0]
            if isinstance(first_item, dict):
                return first_item.get("canvas_height", 512), first_item.get("canvas_width", 512)
        if isinstance(pose_data, dict):
            return pose_data.get("canvas_height", 512), pose_data.get("canvas_width", 512)
        return 512, 512

    def _generate_pose_image(self, pose_data, use_gpu=False):
        """
        Generate an image visualization from pose data.
        Out-of-canvas keypoints are zeroed out in a visualization copy before drawing.
        """
        print(f"\n=== Generating Pose Image ===")

        if not DWPOSE_AVAILABLE:
            print("WARNING: DWPose not available, returning blank image")
            return self._create_blank_image()

        try:
            # Extract canvas dimensions and pose data
            canvas_height, canvas_width = self._get_canvas_dims(pose_data)
            print(f"Canvas size: {canvas_width}x{canvas_height}")

            # Create a copy for visualization to avoid modifying original
            visualization_data = copy.deepcopy(pose_data)

            # Zero-out out-of-canvas keypoints in visualization copy
            self._zero_out_of_canvas(visualization_data, canvas_height, canvas_width)

            # Decode poses from JSON format
            poses, _, _, _ = decode_json_as_poses(visualization_data[0] if isinstance(visualization_data, list) else visualization_data)
            print(f"Decoded {len(poses)} poses from data")

            # Draw poses on canvas (skips missing keypoints because they are zeroed)
            canvas = draw_poses(
                poses,
                canvas_height,
                canvas_width,
                draw_body=True,
                draw_hand=True,
                draw_face=True
            )
            print(f"Drew poses on canvas")

            # Convert numpy array to torch tensor format (normalize to 0-1)
            image_tensor = canvas.astype(np.float32) / 255.0
            image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
            device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
            image_tensor = torch.from_numpy(image_tensor).to(device)  # Convert to torch tensor

            print(f"Converted to tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}, device={image_tensor.device}")
            print("=== Image Generation Complete ===\n")
            return image_tensor

        except Exception as e:
            print(f"ERROR generating pose image: {e}")
            import traceback
            traceback.print_exc()
            return self._create_blank_image()

    
    def _create_blank_image(self, width=512, height=512):
        """Create a blank image tensor in ComfyUI format (CPU)."""
        print(f"Creating blank image {width}x{height}")
        # Return as torch tensor on CPU: (batch, height, width, channels) with values 0-1
        blank = torch.zeros((1, height, width, 3), dtype=torch.float32, device='cpu')
        return blank


NODE_CLASS_MAPPINGS = {"DWPoseRestorator": DwRestorator}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseRestorator": "DWPose Restoration"}
