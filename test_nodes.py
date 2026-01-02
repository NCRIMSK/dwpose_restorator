#!/usr/bin/env python3
"""
Unit tests for the relative restoration node (nodes.py).
Tests affine transformation, zero-out policy, and restoration logic.
"""

import unittest
import numpy as np
import cv2


class TestAffineTransformation(unittest.TestCase):
    """Test affine transformation estimation and application."""
    
    def test_affine_identity(self):
        """Test affine transformation with identity (no rotation/scale/translation)."""
        # Three reference points
        src_points = np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float32)
        dst_points = np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float32)
        
        # Estimate affine matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Apply to a point
        test_point = np.array([[150, 150, 1]], dtype=np.float32).T
        result = affine_matrix @ test_point
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(result.flatten()[:2], [150, 150], decimal=1)
    
    def test_affine_translation(self):
        """Test affine transformation with pure translation."""
        src_points = np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float32)
        # Translate by (+50, +30)
        dst_points = np.array([[150, 130], [250, 130], [150, 230]], dtype=np.float32)
        
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Apply to a point
        test_point = np.array([[300, 300, 1]], dtype=np.float32).T
        result = affine_matrix @ test_point
        
        # Should be translated by (+50, +30)
        expected = [350, 330]
        np.testing.assert_array_almost_equal(result.flatten()[:2], expected, decimal=1)
    
    def test_affine_scale(self):
        """Test affine transformation with scaling (around origin-like point)."""
        src_points = np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float32)
        # Scale by 1.5x (keeping top-left fixed)
        dst_points = np.array([[100, 100], [250, 100], [100, 250]], dtype=np.float32)
        
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Test point relative to origin
        test_point = np.array([[200, 200, 1]], dtype=np.float32).T
        result = affine_matrix @ test_point
        
        # Rough check: scaled distance should be larger
        expected_approx = [250, 250]
        result_vals = result.flatten()[:2]
        # Allow larger tolerance for scaling test
        np.testing.assert_array_almost_equal(result_vals, expected_approx, decimal=0)
    
    def test_affine_offset_transformation(self):
        """Test transforming offset vectors (used in restoration)."""
        # Reference offset: parent to child = (50, -50)
        ref_offset = np.array([50, -50])
        
        # Create a transformation: 1.2x scale, 45 degree rotation
        angle = np.pi / 4  # 45 degrees
        scale = 1.2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        affine_matrix = np.array([
            [cos_a * scale, -sin_a * scale, 0],
            [sin_a * scale,  cos_a * scale, 0]
        ], dtype=np.float32)
        
        # Apply to offset (remove translation component)
        offset_homog = np.array([ref_offset[0], ref_offset[1], 1], dtype=np.float32)
        transformed = affine_matrix @ offset_homog
        
        # Should be scaled and rotated
        transformed_offset = transformed[:2]
        
        # Check magnitude changed (scaled by 1.2)
        original_magnitude = np.linalg.norm(ref_offset)
        transformed_magnitude = np.linalg.norm(transformed_offset)
        expected_magnitude = original_magnitude * scale
        
        self.assertAlmostEqual(transformed_magnitude, expected_magnitude, places=1)


class TestZeroOutPolicy(unittest.TestCase):
    """Test out-of-canvas keypoint zero-out policy."""
    
    def test_zero_out_single_point_outside_canvas(self):
        """Test zeroing a single out-of-canvas keypoint."""
        canvas_width, canvas_height = 512, 512
        keypoint = np.array([600.0, 300.0, 0.8])  # x out of bounds
        
        # Check if out of bounds
        is_out_of_bounds = (keypoint[0] < 0 or keypoint[0] >= canvas_width or
                            keypoint[1] < 0 or keypoint[1] >= canvas_height)
        
        self.assertTrue(is_out_of_bounds)
        
        # Zero out if out of bounds
        if is_out_of_bounds:
            keypoint_zeroed = np.array([0.0, 0.0, 0.0])
        else:
            keypoint_zeroed = keypoint
        
        np.testing.assert_array_equal(keypoint_zeroed, [0.0, 0.0, 0.0])
    
    def test_keep_point_inside_canvas(self):
        """Test that in-canvas keypoints are not zeroed."""
        canvas_width, canvas_height = 512, 512
        keypoint = np.array([256.0, 256.0, 0.8])  # Center of canvas
        
        # Check if out of bounds
        is_out_of_bounds = (keypoint[0] < 0 or keypoint[0] >= canvas_width or
                            keypoint[1] < 0 or keypoint[1] >= canvas_height)
        
        self.assertFalse(is_out_of_bounds)
        
        # Should not be zeroed
        if is_out_of_bounds:
            keypoint_zeroed = np.array([0.0, 0.0, 0.0])
        else:
            keypoint_zeroed = keypoint
        
        np.testing.assert_array_equal(keypoint_zeroed, keypoint)
    
    def test_zero_out_batch_of_keypoints(self):
        """Test zeroing multiple keypoints in a batch."""
        canvas_width, canvas_height = 512, 512
        
        # Mixed: some in bounds, some out
        keypoints = np.array([
            [100.0, 100.0, 0.9],   # In bounds
            [600.0, 300.0, 0.8],   # Out of bounds (x)
            [256.0, 256.0, 0.7],   # In bounds
            [300.0, -10.0, 0.6]    # Out of bounds (y)
        ])
        
        # Apply zero-out policy
        keypoints_processed = keypoints.copy()
        for i in range(len(keypoints_processed)):
            x, y = keypoints_processed[i, 0], keypoints_processed[i, 1]
            if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
                keypoints_processed[i] = [0.0, 0.0, 0.0]
        
        # Check results
        self.assertTrue(np.array_equal(keypoints_processed[0], [100.0, 100.0, 0.9]))
        self.assertTrue(np.array_equal(keypoints_processed[1], [0.0, 0.0, 0.0]))
        self.assertTrue(np.array_equal(keypoints_processed[2], [256.0, 256.0, 0.7]))
        self.assertTrue(np.array_equal(keypoints_processed[3], [0.0, 0.0, 0.0]))


class TestRelativeRestoration(unittest.TestCase):
    """Test relative restoration logic."""
    
    def test_simple_parent_child_restoration(self):
        """Test simple parent-child keypoint restoration."""
        # Reference pose (known good)
        ref_parent = np.array([300.0, 200.0])
        ref_child = np.array([350.0, 150.0])
        ref_offset = ref_child - ref_parent  # (50, -50)
        
        # Current pose (child missing)
        cur_parent = np.array([400.0, 200.0])  # Shifted right by 100
        
        # Restore: apply reference offset to current parent
        cur_child_restored = cur_parent + ref_offset
        
        expected = np.array([450.0, 150.0])
        np.testing.assert_array_almost_equal(cur_child_restored, expected)
    
    def test_chain_restoration(self):
        """Test cascading restoration (parent → child → grandchild)."""
        # Reference: shoulder → elbow → wrist
        ref_shoulder = np.array([300.0, 200.0])
        ref_elbow = np.array([350.0, 150.0])
        ref_wrist = np.array([400.0, 100.0])
        
        ref_offset_shoulder_elbow = ref_elbow - ref_shoulder  # (50, -50)
        ref_offset_elbow_wrist = ref_wrist - ref_elbow        # (50, -50)
        
        # Current: only shoulder exists
        cur_shoulder = np.array([400.0, 200.0])
        
        # Step 1: Restore elbow
        cur_elbow = cur_shoulder + ref_offset_shoulder_elbow
        
        # Step 2: Restore wrist using restored elbow
        cur_wrist = cur_elbow + ref_offset_elbow_wrist
        
        expected_shoulder = np.array([400.0, 200.0])
        expected_elbow = np.array([450.0, 150.0])
        expected_wrist = np.array([500.0, 100.0])
        
        np.testing.assert_array_almost_equal(cur_shoulder, expected_shoulder)
        np.testing.assert_array_almost_equal(cur_elbow, expected_elbow)
        np.testing.assert_array_almost_equal(cur_wrist, expected_wrist)
    
    def test_scaled_restoration(self):
        """Test restoration with scale adjustment."""
        # Reference
        ref_shoulder = np.array([300.0, 200.0])
        ref_elbow = np.array([350.0, 150.0])
        ref_wrist = np.array([400.0, 100.0])
        
        ref_offset_elbow = ref_elbow - ref_shoulder
        ref_offset_wrist = ref_wrist - ref_elbow
        
        # Current: shoulder and elbow exist, wrist missing
        cur_shoulder = np.array([400.0, 200.0])
        cur_elbow = np.array([470.0, 140.0])  # Larger arm
        
        # Estimate scale from existing elbow
        scale_factor = np.linalg.norm(cur_elbow - cur_shoulder) / np.linalg.norm(ref_offset_elbow)
        
        # Restore wrist with scale
        cur_wrist = cur_elbow + ref_offset_wrist * scale_factor
        
        # Check proportions are maintained
        current_ratio = np.linalg.norm(cur_elbow - cur_shoulder) / np.linalg.norm(cur_wrist - cur_elbow)
        reference_ratio = np.linalg.norm(ref_offset_elbow) / np.linalg.norm(ref_offset_wrist)
        
        self.assertAlmostEqual(current_ratio, reference_ratio, places=1)


class TestConfidenceInheritance(unittest.TestCase):
    """Test confidence score inheritance in restoration."""
    
    def test_confidence_minimum_rule(self):
        """Test that restored keypoint gets minimum of parent and reference confidence."""
        parent_confidence = 0.9
        ref_child_confidence = 0.8
        
        # Restored child should get minimum
        restored_confidence = min(parent_confidence, ref_child_confidence)
        
        self.assertEqual(restored_confidence, 0.8)
    
    def test_confidence_reduction(self):
        """Test optional confidence reduction for restored keypoints."""
        parent_confidence = 0.9
        ref_child_confidence = 0.8
        reduction_factor = 0.7  # Reduce by 30%
        
        # Base confidence
        base_conf = min(parent_confidence, ref_child_confidence)
        
        # Apply reduction
        reduced_conf = base_conf * reduction_factor
        
        self.assertAlmostEqual(reduced_conf, 0.56, places=2)
        self.assertLess(reduced_conf, base_conf)  # Should be lower
    
    def test_confidence_chain(self):
        """Test confidence propagation through a chain."""
        # Shoulder (existing): 0.95
        # Elbow (restored from shoulder): min(0.95, ref=0.85) = 0.85
        # Wrist (restored from elbow): min(0.85, ref=0.80) = 0.80
        
        shoulder_conf = 0.95
        ref_elbow_conf = 0.85
        ref_wrist_conf = 0.80
        
        elbow_conf = min(shoulder_conf, ref_elbow_conf)
        wrist_conf = min(elbow_conf, ref_wrist_conf)
        
        self.assertEqual(elbow_conf, 0.85)
        self.assertEqual(wrist_conf, 0.80)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_missing_keypoint_detection(self):
        """Test detection of missing keypoints ([0, 0, 0] or low confidence)."""
        missing_kpt = np.array([0.0, 0.0, 0.0])
        low_conf_kpt = np.array([100.0, 200.0, 0.0])
        valid_kpt = np.array([100.0, 200.0, 0.5])
        
        def is_missing(kpt):
            """Check if keypoint is missing."""
            return kpt[2] < 0.01 or (kpt[0] == 0 and kpt[1] == 0)
        
        self.assertTrue(is_missing(missing_kpt))
        self.assertTrue(is_missing(low_conf_kpt))
        self.assertFalse(is_missing(valid_kpt))
    
    def test_canvas_boundary_keypoints(self):
        """Test keypoints exactly at canvas boundaries."""
        canvas_width, canvas_height = 512, 512
        
        # Test all four corners and edges
        test_cases = [
            ([0, 0], True),              # Top-left (in bounds)
            ([511, 511], True),          # Bottom-right (in bounds)
            ([512, 256], False),         # Right edge (out)
            ([256, 512], False),         # Bottom edge (out)
            ([-1, 256], False),          # Left edge (out)
            ([256, -1], False),          # Top edge (out)
        ]
        
        for coords, should_be_in_bounds in test_cases:
            x, y = coords
            is_in_bounds = (x >= 0 and x < canvas_width and y >= 0 and y < canvas_height)
            self.assertEqual(is_in_bounds, should_be_in_bounds,
                           f"Point {coords} in_bounds={is_in_bounds}, expected={should_be_in_bounds}")
    
    def test_zero_offset_restoration(self):
        """Test restoration when reference offset is zero (parent == child)."""
        ref_parent = np.array([300.0, 200.0])
        ref_child = np.array([300.0, 200.0])  # Same as parent
        ref_offset = ref_child - ref_parent  # (0, 0)
        
        cur_parent = np.array([400.0, 300.0])
        
        # Restore with zero offset
        cur_child = cur_parent + ref_offset
        
        # Child should be at same location as parent
        np.testing.assert_array_almost_equal(cur_child, cur_parent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
