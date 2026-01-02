#!/usr/bin/env python3
"""
Test and demonstration script for the relative restoration implementation.
This shows the key concepts with simple examples.
"""

import numpy as np
import cv2

# Simulate keypoint restoration logic
def demonstrate_relative_restoration():
    print("=" * 70)
    print("DWPose Relative Restoration - Concept Demonstration")
    print("=" * 70)
    
    # Example 1: Simple parent-child restoration
    print("\n[EXAMPLE 1] Simple Parent-Child Restoration")
    print("-" * 70)
    
    # Reference pose (known good positions)
    ref_shoulder = np.array([300.0, 200.0])
    ref_elbow = np.array([350.0, 150.0])
    ref_wrist = np.array([400.0, 100.0])
    
    ref_offset_elbow = ref_elbow - ref_shoulder  # (50, -50)
    ref_offset_wrist = ref_wrist - ref_elbow      # (50, -50)
    
    print(f"Reference Pose:")
    print(f"  Shoulder: {ref_shoulder}")
    print(f"  Elbow:    {ref_elbow}")
    print(f"  Wrist:    {ref_wrist}")
    print(f"Reference Offsets:")
    print(f"  Shoulder→Elbow: {ref_offset_elbow}")
    print(f"  Elbow→Wrist:    {ref_offset_wrist}")
    
    # Current pose (some keypoints missing)
    cur_shoulder = np.array([400.0, 200.0])  # Shifted right
    cur_elbow = None  # MISSING
    cur_wrist = None  # MISSING
    
    print(f"\nCurrent Pose (partial):")
    print(f"  Shoulder: {cur_shoulder}")
    print(f"  Elbow:    MISSING")
    print(f"  Wrist:    MISSING")
    
    # Estimate scale/rotation from existing points
    # In this simple case, shoulder shifted right by 100 pixels
    scale_factor = 1.0  # No scale change
    
    # Restore elbow
    cur_elbow = cur_shoulder + ref_offset_elbow * scale_factor
    cur_wrist = cur_elbow + ref_offset_wrist * scale_factor
    
    print(f"\nRestored Pose:")
    print(f"  Shoulder: {cur_shoulder}")
    print(f"  Elbow:    {cur_elbow} (restored)")
    print(f"  Wrist:    {cur_wrist} (restored)")
    print(f"\n✓ Proportions maintained!")
    
    # Example 2: Restoration with scaling
    print("\n\n[EXAMPLE 2] Restoration with Scale Change")
    print("-" * 70)
    
    cur_shoulder2 = np.array([400.0, 200.0])
    cur_elbow2 = np.array([470.0, 140.0])  # Exists (larger arm)
    cur_wrist2 = None  # MISSING
    
    # Estimate scale from existing elbow
    # Reference: shoulder→elbow = (50, -50)
    # Current: shoulder→elbow = (70, -60)
    # Scale: roughly 1.4x
    scale_factor_2 = np.linalg.norm(cur_elbow2 - cur_shoulder2) / np.linalg.norm(ref_offset_elbow)
    
    print(f"Current Pose:")
    print(f"  Shoulder:    {cur_shoulder2}")
    print(f"  Elbow:       {cur_elbow2} (exists)")
    print(f"  Wrist:       MISSING")
    print(f"\nEstimated scale factor: {scale_factor_2:.2f}")
    
    # Restore wrist with scale factor
    cur_wrist2 = cur_elbow2 + ref_offset_wrist * scale_factor_2
    
    print(f"\nRestored Pose:")
    print(f"  Wrist: {cur_wrist2} (restored with scale {scale_factor_2:.2f})")
    print(f"\n✓ Scaled proportions maintained!")
    
    # Example 3: Out-of-canvas handling
    print("\n\n[EXAMPLE 3] Out-of-Canvas Keypoint Handling")
    print("-" * 70)
    
    canvas_width, canvas_height = 512, 512
    
    # Pose with arm extended to edge
    shoulder3 = np.array([450.0, 256.0])
    elbow3 = np.array([500.0, 200.0])
    wrist3 = np.array([600.0, 100.0])  # OUT OF CANVAS!
    
    print(f"Canvas: {canvas_width}x{canvas_height}")
    print(f"\nPose with out-of-canvas keypoint:")
    print(f"  Shoulder: {shoulder3}")
    print(f"  Elbow:    {elbow3}")
    print(f"  Wrist:    {wrist3}")
    
    # For internal calculations, keep as-is
    print(f"\nInternal calculations: Use unclamped coordinates {wrist3}")
    
    # For visualization/export, zero out out-of-canvas keypoints (mark as missing)
    wrist3_exported = np.array([0.0, 0.0, 0.0]) if (wrist3[0] < 0 or wrist3[0] >= canvas_width or 
                                                       wrist3[1] < 0 or wrist3[1] >= canvas_height) else wrist3
    print(f"For visualization/export: Zero-out to {wrist3_exported} (marked as missing)")
    print(f"\n✓ Internal precision maintained, out-of-canvas keypoints marked as missing in outputs!")
    
    # Example 4: Affine transformation concept
    print("\n\n[EXAMPLE 4] Affine Transformation (Rotation + Scale + Translation)")
    print("-" * 70)
    
    # Create a simple rotation matrix (45 degrees)
    angle = np.pi / 4  # 45 degrees
    scale = 1.2
    tx, ty = 50, 30  # translation
    
    # Build affine matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    affine = np.array([
        [cos_a * scale, -sin_a * scale, tx],
        [sin_a * scale,  cos_a * scale, ty]
    ], dtype=np.float32)
    
    print(f"Affine matrix (45° rotation, 1.2x scale, +50,+30 translation):")
    print(affine)
    
    # Apply to an offset vector
    offset = np.array([50, -50])
    offset_homog = np.array([offset[0], offset[1], 1])
    transformed = affine @ offset_homog
    
    print(f"\nApplying to offset (50, -50):")
    print(f"  Original:    {offset}")
    print(f"  Transformed: {transformed[:2]}")
    print(f"\n✓ Offset accounts for rotation, scale, and translation!")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Relative restoration preserves proportions across pose variations")
    print("2. Affine transformation captures rotation, scale, and translation")
    print("3. Hierarchical parent-child approach enables chain restoration")
    print("4. Out-of-canvas handling keeps precision internally")
    print("5. Confidence scoring allows downstream weighting of restored points")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_relative_restoration()
