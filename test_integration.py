#!/usr/bin/env python3
"""
Integration test for the DwRestorator node in ComfyUI environment.
Tests the node with realistic pose data and ComfyUI workflow simulation.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add current directory and parent to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Import the node
from nodes import DwRestorator


def create_test_pose(num_people=1, width=512, height=512):
    """Create a realistic DWPose JSON structure for testing."""
    pose_data = {
        "people": []
    }
    
    for person_idx in range(num_people):
        person = {
            "person_id": [person_idx],
            # Body keypoints: 18 points (x, y, confidence)
            "pose_keypoints_2d": [
                # Head region
                256.0 + person_idx * 50, 100.0, 0.95,   # 0: Nose
                240.0 + person_idx * 50, 90.0, 0.92,    # 1: Left Eye
                272.0 + person_idx * 50, 90.0, 0.92,    # 2: Right Eye
                230.0 + person_idx * 50, 110.0, 0.85,   # 3: Left Ear
                282.0 + person_idx * 50, 110.0, 0.85,   # 4: Right Ear
                # Shoulder to Elbow to Wrist (Left)
                220.0 + person_idx * 50, 180.0, 0.90,   # 5: Left Shoulder
                180.0 + person_idx * 50, 250.0, 0.88,   # 6: Left Elbow
                0.0, 0.0, 0.0,                           # 7: Left Wrist (MISSING)
                # Shoulder to Elbow to Wrist (Right)
                292.0 + person_idx * 50, 180.0, 0.90,   # 8: Right Shoulder
                332.0 + person_idx * 50, 250.0, 0.88,   # 9: Right Elbow
                380.0 + person_idx * 50, 320.0, 0.85,   # 10: Right Wrist
                # Hips
                240.0 + person_idx * 50, 320.0, 0.89,   # 11: Left Hip
                272.0 + person_idx * 50, 320.0, 0.89,   # 12: Right Hip
                # Knees
                230.0 + person_idx * 50, 420.0, 0.87,   # 13: Left Knee
                282.0 + person_idx * 50, 420.0, 0.87,   # 14: Right Knee
                # Ankles
                220.0 + person_idx * 50, 500.0, 0.85,   # 15: Left Ankle
                292.0 + person_idx * 50, 500.0, 0.85,   # 16: Right Ankle
                # Neck
                256.0 + person_idx * 50, 140.0, 0.90,   # 17: Neck
            ],
            # Hand keypoints: 21 per hand (left and right)
            "hand_left_keypoints_2d": [0.0] * 63,  # 21 points * 3
            "hand_right_keypoints_2d": [0.0] * 63,
            # Face keypoints: 70 points
            "face_keypoints_2d": [0.0] * 210,  # 70 points * 3
        }
        pose_data["people"].append(person)
    
    return pose_data


def create_reference_pose(width=512, height=512):
    """Create a reference pose (complete, good quality)."""
    pose_data = {
        "people": [
            {
                "person_id": [0],
                # Complete body keypoints
                "pose_keypoints_2d": [
                    # Head region
                    256.0, 100.0, 0.98,   # 0: Nose
                    240.0, 90.0, 0.96,    # 1: Left Eye
                    272.0, 90.0, 0.96,    # 2: Right Eye
                    230.0, 110.0, 0.90,   # 3: Left Ear
                    282.0, 110.0, 0.90,   # 4: Right Ear
                    # Shoulder to Elbow to Wrist (Left)
                    220.0, 180.0, 0.95,   # 5: Left Shoulder
                    170.0, 260.0, 0.94,   # 6: Left Elbow
                    120.0, 340.0, 0.92,   # 7: Left Wrist
                    # Shoulder to Elbow to Wrist (Right)
                    292.0, 180.0, 0.95,   # 8: Right Shoulder
                    342.0, 260.0, 0.94,   # 9: Right Elbow
                    392.0, 340.0, 0.92,   # 10: Right Wrist
                    # Hips
                    240.0, 320.0, 0.93,   # 11: Left Hip
                    272.0, 320.0, 0.93,   # 12: Right Hip
                    # Knees
                    230.0, 420.0, 0.91,   # 13: Left Knee
                    282.0, 420.0, 0.91,   # 14: Right Knee
                    # Ankles
                    220.0, 500.0, 0.90,   # 15: Left Ankle
                    292.0, 500.0, 0.90,   # 16: Right Ankle
                    # Neck
                    256.0, 140.0, 0.95,   # 17: Neck
                ],
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63,
                "face_keypoints_2d": [0.0] * 210,
            }
        ]
    }
    return pose_data


def test_node_initialization():
    """Test that the node initializes correctly."""
    print("\n[TEST 1] Node Initialization")
    print("-" * 70)
    
    try:
        node = DwRestorator()
        print("✓ Node initialized successfully")
        
        # Check INPUT_TYPES
        input_types = node.INPUT_TYPES()
        required_inputs = input_types.get("required", {})
        optional_inputs = input_types.get("optional", {})
        
        assert "pose_keypoints" in required_inputs, "Missing 'pose_keypoints' in required inputs"
        print("✓ Required inputs defined correctly")
        
        assert "reduce_confidence" in optional_inputs, "Missing 'reduce_confidence' in optional inputs"
        assert "use_gpu" in optional_inputs, "Missing 'use_gpu' in optional inputs"
        print("✓ Optional inputs defined correctly")
        
        # Check RETURN_TYPES (it's a tuple, not callable)
        return_types = node.RETURN_TYPES
        assert return_types[0] == "IMAGE", "First return type should be IMAGE"
        print("✓ Return types defined correctly")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_with_missing_keypoints():
    """Test node with missing keypoints (main use case)."""
    print("\n[TEST 2] Restoration with Missing Keypoints")
    print("-" * 70)
    
    try:
        node = DwRestorator()
        
        # Create test data
        pose_with_gaps = create_test_pose(num_people=1)
        ref_pose = create_reference_pose()
        
        # Check that left wrist is missing in test pose
        test_pose_data = pose_with_gaps["people"][0]["pose_keypoints_2d"]
        left_wrist_idx = 7 * 3  # Wrist is keypoint 7
        assert test_pose_data[left_wrist_idx:left_wrist_idx+3] == [0.0, 0.0, 0.0], "Test pose should have missing left wrist"
        print("✓ Test pose has missing left wrist as expected")
        
        # Run restoration (pass dicts, not JSON strings)
        image_tensor, restored_pose = node.dwrestore(
            pose_keypoints=pose_with_gaps,
            ref_pose=ref_pose,
            reduce_confidence=True,
            confidence_reduction_factor=0.7,
            use_gpu=False
        )
        
        # Check output types
        assert image_tensor is not None, "Image tensor should not be None"
        print(f"✓ Image tensor generated: shape {image_tensor.shape}")
        
        # restored_pose should be a dict or JSON string, check both
        if isinstance(restored_pose, str):
            restored_data = json.loads(restored_pose)
        else:
            restored_data = restored_pose
        
        assert "people" in restored_data, "Restored pose should have 'people' key"
        print("✓ Restored pose is valid")
        
        # Check that left wrist was restored
        restored_pose_data = restored_data["people"][0]["pose_keypoints_2d"]
        restored_wrist = restored_pose_data[left_wrist_idx:left_wrist_idx+3]
        assert restored_wrist[2] > 0, f"Restored wrist should have confidence > 0, got {restored_wrist[2]}"
        print(f"✓ Left wrist restored: [{restored_wrist[0]:.1f}, {restored_wrist[1]:.1f}, {restored_wrist[2]:.2f}]")
        
        # Check that in-canvas restored points are preserved
        assert restored_wrist[0] > 0 and restored_wrist[1] > 0, "Restored wrist should have valid coordinates"
        print(f"✓ Restored coordinates are valid (in-canvas)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_out_of_canvas_handling():
    """Test that out-of-canvas keypoints are handled correctly."""
    print("\n[TEST 3] Out-of-Canvas Keypoint Handling")
    print("-" * 70)
    
    try:
        # Create pose with out-of-canvas keypoint
        pose_data = {
            "people": [
                {
                    "person_id": [0],
                    # Minimal body keypoints for testing
                    "pose_keypoints_2d": [
                        # Some keypoints
                        256.0, 100.0, 0.95,    # 0: Nose
                        220.0, 180.0, 0.90,    # 5: Left Shoulder
                        600.0, 250.0, 0.85,    # 6: Out-of-canvas X
                        # Rest are zeros
                    ] + [0.0] * (18*3 - 9),
                    "hand_left_keypoints_2d": [0.0] * 63,
                    "hand_right_keypoints_2d": [0.0] * 63,
                    "face_keypoints_2d": [0.0] * 210,
                }
            ]
        }
        
        # Run without reference (should handle gracefully)
        node = DwRestorator()
        
        # Create a minimal reference pose
        ref_pose = {
            "people": [
                {
                    "person_id": [0],
                    "pose_keypoints_2d": [0.0] * 54,  # 18 keypoints * 3
                    "hand_left_keypoints_2d": [0.0] * 63,
                    "hand_right_keypoints_2d": [0.0] * 63,
                    "face_keypoints_2d": [0.0] * 210,
                }
            ]
        }
        
        image_tensor, restored_pose = node.dwrestore(
            pose_keypoints=pose_data,
            ref_pose=ref_pose,
            use_gpu=False
        )
        
        # Check that output is valid
        assert image_tensor is not None, "Should generate image even with out-of-canvas points"
        print("✓ Image generated despite out-of-canvas keypoints")
        
        if isinstance(restored_pose, str):
            restored_data = json.loads(restored_pose)
        else:
            restored_data = restored_pose
        
        # Check that output is valid
        assert "people" in restored_data, "Should have people key"
        print("✓ Out-of-canvas handling completed without errors")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_fallback():
    """Test that use_gpu flag falls back to CPU gracefully."""
    print("\n[TEST 4] GPU Fallback (use_gpu=True)")
    print("-" * 70)
    
    try:
        node = DwRestorator()
        pose_data = create_test_pose(num_people=1)
        
        # Create minimal reference
        ref_pose = {
            "people": [{
                "person_id": [0],
                "pose_keypoints_2d": [0.0] * 54,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63,
                "face_keypoints_2d": [0.0] * 210,
            }]
        }
        
        # Try with use_gpu=True (should fall back to CPU if no CUDA)
        image_tensor, restored_pose = node.dwrestore(
            pose_keypoints=pose_data,
            ref_pose=ref_pose,
            use_gpu=True  # Request GPU but should fallback
        )
        
        assert image_tensor is not None, "Should work even if GPU not available"
        print("✓ GPU flag handled correctly (CPU fallback if needed)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_people():
    """Test with multiple people in one frame."""
    print("\n[TEST 5] Multiple People in Frame")
    print("-" * 70)
    
    try:
        node = DwRestorator()
        pose_data = create_test_pose(num_people=2)  # Two people
        ref_pose = create_reference_pose()
        
        # Run restoration (pass dicts, not JSON strings)
        image_tensor, restored_pose = node.dwrestore(
            pose_keypoints=pose_data,
            ref_pose=ref_pose,
            use_gpu=False
        )
        
        if isinstance(restored_pose, str):
            restored_data = json.loads(restored_pose)
        else:
            restored_data = restored_pose
        
        assert len(restored_data["people"]) == 2, "Should process both people"
        print(f"✓ Processed {len(restored_data['people'])} people successfully")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("DwRestorator Node - Integration Tests")
    print("=" * 70)
    
    tests = [
        test_node_initialization,
        test_node_with_missing_keypoints,
        test_out_of_canvas_handling,
        test_gpu_fallback,
        test_multiple_people,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
