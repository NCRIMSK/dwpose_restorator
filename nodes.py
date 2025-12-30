import copy

class DwRestorator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pose_keypoints": ("POSE_KEYPOINT",)},
            "optional": {"ref_pose": ("POSE_KEYPOINT",)},
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_restored",)
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
        if ref_pose is None:
            return (pose_keypoints,)

        out = copy.deepcopy(pose_keypoints)
        ref = ref_pose  # не копируем — только читаем

    
        def get_person0(x):
            # x может быть dict с "people"
            if isinstance(x, dict) and "people" in x and isinstance(x["people"], list) and x["people"]:
                return x["people"][0], "dict_people0"
        
            # x может быть list людей (people list)
            if isinstance(x, list) and x and isinstance(x[0], dict):
                return x[0], "list_people0"
        
            raise TypeError(f"Unsupported POSE_KEYPOINT structure: {type(x)}")

        pin,  mode_in  = get_person0(out)
        pref, mode_ref = get_person0(ref)

        keys: list[str] = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]

        for k in keys:
            vin = pin.get(k, None)
            vref = pref.get(k, None)    
            if vin is None and vref is not None:
                pin[k] = copy.deepcopy(vref)
                continue
            if isinstance(vin, list) and isinstance(vref, list):
                pin[k] = self._repair_triplets(vin, vref)
            elif vin is not None and vref is not None and (not isinstance(vin, list) or not isinstance(vref, list)):
                # Skip keys with unexpected types
                print(f"Warning: Skipping key '{k}' - expected list, got vin={type(vin).__name__}, vref={type(vref).__name__}")

        return (out,)


NODE_CLASS_MAPPINGS = {"DWPoseRestorator": DwRestorator}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseRestorator": "DWPose Restoration"}
