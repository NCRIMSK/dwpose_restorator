import copy
import numpy as np
import torch
import sys
import os

# Try to import DWPose utilities
try:
    from custom_controlnet_aux.dwpose import (
        decode_json_as_poses,
        draw_poses
    )
    DWPOSE_AVAILABLE = True
    print("[DWRestorator] DWPose utilities loaded successfully")
except ImportError as e:
    print(f"[DWRestorator] WARNING: DWPose utilities not found: {e}")
    print("[DWRestorator] Will use fallback image generation")
    DWPOSE_AVAILABLE = False


class DwRestorator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pose_keypoints": ("POSE_KEYPOINT",)},
            "optional": {"ref_pose": ("POSE_KEYPOINT",)},
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_image", "pose_restored",)
    CATEGORY = "DWPoseRestorator"
    FUNCTION = "dwrestore"

    def _repair_triplets(self, dst, ref):
        # dst/ref: flat list [x,y,c, x,y,c, ...]
        if not isinstance(dst, list) or not isinstance(ref, list):
            return dst
        n = min(len(dst), len(ref))
        for i in range(0, n - (n % 3), 3):
            x, y, c = dst[i], dst[i+1], dst[i+2]
            if x == 0.0 and y == 0.0 and c == 0.0:
                dst[i], dst[i+1], dst[i+2] = ref[i], ref[i+1], ref[i+2]
        return dst

    def dwrestore(self, pose_keypoints, ref_pose=None):
        print(f"\n=== DwRestorator Debug ===")
        print(f"pose_keypoints type: {type(pose_keypoints)}")
        print(f"ref_pose type: {type(ref_pose)}")
        
        if ref_pose is None:
            print("ERROR: ref_pose is None, returning unchanged")
            return (pose_keypoints,)

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
            return (out,)

        keys = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]

        for k in keys:
            vin = pin.get(k, None)
            vref = pref.get(k, None)
            
            print(f"\nProcessing key '{k}':")
            print(f"  vin is None: {vin is None}, vref is None: {vref is None}")
            if vin is not None and isinstance(vin, list):
                print(f"  vin length: {len(vin)}, first 3 values: {vin[:3]}")
            if vref is not None and isinstance(vref, list):
                print(f"  vref length: {len(vref)}, first 3 values: {vref[:3]}")
            
            # If input is None/null and reference has data, restore from reference
            if vin is None and vref is not None:
                print(f"  -> Restoring {k} from reference (was null)")
                pin[k] = copy.deepcopy(vref)
                continue
            
            # If input is None/null but reference is also None/null, skip
            if vin is None and vref is None:
                print(f"  -> Both null, skipping")
                continue
            
            # Both are lists: repair triplets with zero values
            if isinstance(vin, list) and isinstance(vref, list):
                print(f"  -> Repairing triplets")
                original_vin = str(vin[:9])
                pin[k] = self._repair_triplets(vin, vref)
                print(f"     Before: {original_vin}")
                print(f"     After: {str(pin[k][:9])}")
                continue
            
            # Input exists but reference doesn't: keep input as-is
            if vin is not None and vref is None:
                print(f"  -> Input exists but ref is null, keeping input")
                continue
            
            # Unexpected type combinations: log warning
            print(f"  -> WARNING: unexpected types - vin={type(vin).__name__}, vref={type(vref).__name__}")

        print(f"=== Restoration complete ===\n")
        
        # Generate image output if DWPose is available
        image_output = self._generate_pose_image(out)
        
        return (image_output, out,)

    def _generate_pose_image(self, pose_data):
        """
        Generate an image visualization from pose data.
        """
        print(f"\n=== Generating Pose Image ===")
        
        if not DWPOSE_AVAILABLE:
            print("WARNING: DWPose not available, returning blank image")
            return self._create_blank_image()
        
        try:
            # Extract canvas dimensions and pose data
            if isinstance(pose_data, list) and len(pose_data) > 0:
                first_item = pose_data[0]
                if isinstance(first_item, dict):
                    canvas_height = first_item.get("canvas_height", 512)
                    canvas_width = first_item.get("canvas_width", 512)
                    print(f"Canvas size: {canvas_width}x{canvas_height}")
                    
                    # Decode poses from JSON format
                    poses, _, _, _ = decode_json_as_poses(first_item)
                    print(f"Decoded {len(poses)} poses from data")
                    
                    # Draw poses on canvas
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
                    
                    print(f"Converted to tensor: shape={image_tensor.shape}")
                    print("=== Image Generation Complete ===\n")
                    return image_tensor
            
            print("ERROR: Unexpected pose_data format")
            return self._create_blank_image()
            
        except Exception as e:
            print(f"ERROR generating pose image: {e}")
            import traceback
            traceback.print_exc()
            return self._create_blank_image()
    
    def _create_blank_image(self, width=512, height=512):
        """Create a blank image tensor."""
        print(f"Creating blank image {width}x{height}")
        blank = np.zeros((1, height, width, 3), dtype=np.float32)
        return blank


NODE_CLASS_MAPPINGS = {"DWPoseRestorator": DwRestorator}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseRestorator": "DWPose Restoration"}
