"""DWPose Restorator - ComfyUI custom node for restoring corrupted pose keypoints."""
from .nodes import DwRestorator

NODE_CLASS_MAPPINGS = {"DWPoseRestorator": DwRestorator}
NODE_DISPLAY_NAME_MAPPINGS = {"DWPoseRestorator": "DWPose Restoration"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
