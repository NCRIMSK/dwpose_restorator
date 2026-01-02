# AGENTS.md - Complete Project Overview

## Project Summary

**DWPose Restorator** is a ComfyUI custom node that intelligently restores missing or corrupted body/hand/face keypoints from pose detection using a **relative, hierarchical restoration algorithm**. It leverages skeleton structure and affine transformations to maintain anatomically correct proportions even when the pose is at different positions, scales, or rotations compared to a reference pose.

---

## 🎯 Core Functionality

### What It Does
1. **Input:** Current pose (with missing keypoints) + Reference pose (complete/better quality)
2. **Process:** Uses skeleton hierarchy and affine transformation to restore missing keypoints proportionally
3. **Output:** 
   - Restored pose data (JSON-serializable)
   - Visualization image (PIL/torch tensor)

### Key Innovation: Relative Restoration
Instead of copying absolute coordinates:
- Detects existing keypoints in current pose
- For missing keypoints, calculates offset from parent→child in reference
- **Transforms** this offset using an affine matrix (captures rotation, scale, translation)
- Applies transformed offset to current parent position
- Maintains body proportions even at different scales/rotations

---

## 📁 Project Structure

```
dwpose_restorator/
├── Python Backend (ComfyUI Node)
│   ├── __init__.py                         # Package entry point
│   ├── nodes.py                            # Main DwRestorator node (606 lines)
│   ├── pose_types.py                       # Type definitions (NamedTuple structures)
│   ├── pose_visualization.py               # Rendering & decode functions
│   ├── requirements.txt                    # Python dependencies
│   ├── pyproject.toml                      # Build config & metadata
│   └── test_nodes.py                       # Unit tests
│
├── React/TypeScript UI Frontend
│   ├── ui/
│   │   ├── package.json                    # Node.js dependencies
│   │   ├── tsconfig.json                   # TypeScript config
│   │   ├── vite.config.js/ts               # Vite bundler config
│   │   ├── jest.config.js                  # Jest test config
│   │   ├── eslint.config.js                # Linting rules
│   │   ├── src/
│   │   │   ├── App.tsx                     # Main React component
│   │   │   ├── main.tsx                    # React entry point
│   │   │   ├── App.css                     # Styling
│   │   │   ├── utils/
│   │   │   │   └── i18n.ts                 # Internationalization setup
│   │   │   └── __tests__/
│   │   │       └── dummy.test.tsx          # Example test
│   │   ├── public/
│   │   │   └── locales/                    # Translation files
│   │   │       ├── en/main.json            # English (US)
│   │   │       └── zh/main.json            # Chinese (simplified)
│   │   └── dist/                           # Compiled output (built by Vite)
│
└── Documentation & Examples
    ├── README.md                           # Quick start guide
    ├── QUICK_REFERENCE.md                  # Algorithm at a glance
    ├── IMPLEMENTATION_GUIDE.md             # Detailed technical guide
    ├── TECHNICAL_ARCHITECTURE.md           # Deep dive design docs
    ├── COMPLETION_REPORT.md                # Implementation status
    ├── INDEPENDENCE_REPORT.md              # Dependency removal details
    ├── CHANGES.md                          # Detailed changelog
    ├── VISUAL_OVERVIEW.md                  # Diagrams & illustrations
    ├── DOCUMENTATION_INDEX.md              # Navigation guide
    ├── AGENTS.md                           # This file
    ├── demonstration.py                    # Runnable code examples
    └── LICENSE                             # GNU GPLv3
```

---

## 🔧 Core Technology Stack

### Backend (Python)
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | ≥3.10 | Runtime |
| NumPy | ≥1.24 | Array operations, matrix math |
| OpenCV | ≥4.7.0 | Image processing, affine transforms |
| PyTorch | ≥2.1.0 | Tensor operations, GPU support |
| Pillow | ≥9.0.0 | Image I/O |

### Frontend (React/TypeScript)
| Component | Version | Purpose |
|-----------|---------|---------|
| React | ^18.2.0 | UI framework |
| TypeScript | ^5.4.2 | Type-safe JavaScript |
| Vite | ^5.2.10 | Fast build bundler |
| React-i18next | ^14.1.0 | Internationalization |
| Jest | ^29.7.0 | Unit testing |
| ESLint | ^9.27.0 | Code linting |

### ComfyUI Integration
- **Type Definitions:** `@comfyorg/comfyui-frontend-types` (v1.20.2)
- **Custom Node Format:** Standard ComfyUI extension pattern

---

## 📊 Algorithm Overview

### Main Workflow (in `DwRestorator.dwrestore()`)

```
1. INPUT VALIDATION
   ├─ Check ref_pose exists
   ├─ Extract person dict from nested structure
   └─ Validate keypoint formats

2. BODY RESTORATION
   ├─ Extract keypoints from flat list → tuples
   ├─ Estimate affine transform (existing keypoints)
   ├─ Restore body keypoints using BODY_HIERARCHY
   └─ Convert back to flat list with native Python floats

3. HAND RESTORATION (Left & Right)
   ├─ Extract 21 hand keypoints
   ├─ Estimate affine transform per hand
   ├─ Restore using HAND_HIERARCHY (palm-based fingers)
   └─ Convert with Python floats

4. FACE RESTORATION
   ├─ Extract 68-70 face landmarks
   ├─ Use local hierarchy (closest neighbor as anchor)
   ├─ Restore without global skeleton
   └─ Convert with Python floats

5. EXPORT PREPARATION
   ├─ Deep-copy pose data
   ├─ Zero-out out-of-canvas keypoints for visualization
   ├─ Recursively convert all numpy types → Python types (JSON safe)
   └─ Return: (image_tensor, restored_pose_dict)

6. IMAGE GENERATION
   ├─ Decode poses from JSON format
   ├─ Draw skeletons on canvas (768x1365, configurable)
   ├─ Convert to torch tensor (0-1 normalized)
   └─ Return for visualization
```

### Affine Transformation

```
Purpose: Detect how pose changed (rotation, scale, translation)

Algorithm:
1. Find keypoints existing in BOTH current & reference (conf > 0.3)
2. Need minimum 3 pairs:
   - Option A: cv2.getAffineTransform(src[:3], dst[:3])
   - Option B: np.linalg.lstsq for >3 points (more robust)
3. Result: 2×3 transformation matrix
4. Apply to offset vectors (not absolute coordinates!)

Matrix:
  [a  b  tx]   [x]   [x']
  [c  d  ty] × [y] = [y']
               [1]

Key: Remove translation when transforming offsets (offset is relative)
```

### Skeleton Hierarchies

#### Body (18 OpenPose COCO keypoints)
```
           Nose(0)
           /     \
      L.Eye(1)   R.Eye(2)
           |          |
      L.Ear(3)   R.Ear(4)
      
           Neck(17)
          /         \
    L.Shoulder(5)  R.Shoulder(6)
         |               |
    L.Elbow(7)     R.Elbow(8)
         |               |
    L.Wrist(9)     R.Wrist(10)
    
    L.Hip(11)  R.Hip(12)
         |           |
    L.Knee(13) R.Knee(14)
         |           |
    L.Ankle(15) R.Ankle(16)
```

#### Hand (21 keypoints per hand)
```
        Palm(0)
        /  |  \  \  \
    T(1) I(5) M(9) R(13) P(17)
    |     |    |     |     |
    2     6    10    14    18
    |     |    |     |     |
    3     7    11    15    19
    |     |    |     |     |
    4     8    12    16    20
```

#### Face (68-70 landmarks)
- No fixed hierarchy
- Uses **local/closest neighbor restoration**
- Finds nearest existing keypoint as anchor

### Keypoint Missing Detection
```python
def _is_keypoint_missing(x, y, confidence):
    return x == 0.0 and y == 0.0 and confidence == 0.0
```
Matches DWPose convention: missing keypoints are (0, 0, 0)

---

## 🔌 ComfyUI Node Interface

### Node: `DWPoseRestorator`

**Category:** `DWPoseRestorator`  
**Function:** `dwrestore`

### Inputs

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `pose_keypoints` | POSE_KEYPOINT | Required | — | Current pose with missing keypoints |
| `ref_pose` | POSE_KEYPOINT | None | — | Reference pose (complete/better) |
| `reduce_confidence` | BOOLEAN | True | — | Lower confidence of restored points |
| `confidence_reduction_factor` | FLOAT | 0.7 | 0.0–1.0 | Multiplier for restored confidence |
| `use_gpu` | BOOLEAN | False | — | Use GPU for tensor operations |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `pose_image` | IMAGE | RGB visualization (torch tensor) |
| `pose_restored` | POSE_KEYPOINT | Restored pose with completed keypoints |

### Data Format

**POSE_KEYPOINT** structure:
```python
# Input/Output: List or Dict
{
  "people": [
    {
      "pose_keypoints_2d": [x0, y0, c0, x1, y1, c1, ...],  # 18×3 flat
      "hand_left_keypoints_2d": [x0, y0, c0, ...],          # 21×3 flat
      "hand_right_keypoints_2d": [x0, y0, c0, ...],         # 21×3 flat
      "face_keypoints_2d": [x0, y0, c0, ...],               # 68–70×3 flat
      "canvas_height": 512,
      "canvas_width": 512
    }
  ]
}

# Or direct person dict (auto-detected):
{
  "pose_keypoints_2d": [...],
  "hand_left_keypoints_2d": [...],
  "hand_right_keypoints_2d": [...],
  "face_keypoints_2d": [...],
  "canvas_height": 512,
  "canvas_width": 512
}
```

---

## 📋 Key Features & Parameters

### 1. Hierarchical Restoration
- Uses skeleton structure to infer missing keypoints
- Respects parent-child relationships
- Cascading: restores parents first, then children

### 2. Affine Transformation Matching
- Automatically detects pose differences
- Handles rotation, scale, translation
- Maintains proportions across different view angles

### 3. Confidence Tracking
- Option to mark restored points as lower confidence
- Configurable reduction factor (default 0.7×)
- Helps downstream processes identify restored vs. detected keypoints

### 4. Out-of-Canvas Handling
- Keeps internal values for calculations
- Visualization copy zeros out-of-bounds points
- Prevents false detections at image edges

### 5. Flexible Input Parsing
- Handles multiple pose data structures
- Auto-detects nested "people" arrays
- Works with direct person dicts

### 6. Internationalization (i18n)
- UI supports English (en) and Chinese (zh)
- `i18next` for runtime language switching
- Localization files in `ui/public/locales/`

### 7. Independence from controlnet_aux
- Self-contained pose visualization
- Graceful fallback if dependencies missing
- Works in ComfyUI, standalone tests, development

---

## 🛠️ Development & Testing

### Python Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest test_nodes.py -v
```

**Test Files:**
- `test_nodes.py` — Unit tests (16 tests)
- `test_integration.py` — Integration tests (5 tests)
- `demonstration.py` — Runnable examples

### React/TypeScript Development

```bash
cd ui/

# Install dependencies
npm install

# Development (watch mode)
npm run dev           # Vite dev server
npm run watch         # Watch TypeScript + Vite build

# Production build
npm run build

# Testing
npm test              # Run Jest
npm run test:watch    # Watch tests

# Code quality
npm run lint          # Check linting
npm run lint:fix      # Auto-fix issues
npm run format        # Pretty-print code
npm run typecheck     # TypeScript validation
```

### Build & Distribution

```bash
# Build frontend
cd ui/
npm install
npm run build

# The `dist/` folder is what ComfyUI loads
# Python backend is auto-loaded from root directory
```

---

## 🚀 Installation & Deployment

### For ComfyUI Users (Recommended)

1. **Via ComfyUI Manager:**
   - Open Manager → Search "DWPose Restorator"
   - Click Install

2. **Manual Installation:**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/NCRIMSK/dwpose_restorator.git
   cd dwpose_restorator/ui
   npm install && npm run build
   # Restart ComfyUI
   ```

### Environment Setup

**Python Environment:**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- NumPy, OpenCV, PyTorch (CPU or GPU)
- Pillow for image handling

**Node.js/NPM:**
- Node.js 16+ with npm
- Development: TypeScript, Jest, Vite, ESLint

---

## 📚 Documentation Files Map

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Quick start, installation, usage overview | All users |
| **QUICK_REFERENCE.md** | One-page algorithm summary, skeletons, parameters | Busy developers |
| **IMPLEMENTATION_GUIDE.md** | Detailed concept explanations, edge cases | Developers integrating |
| **TECHNICAL_ARCHITECTURE.md** | Deep dive: system design, math, optimization | Advanced developers |
| **COMPLETION_REPORT.md** | What was implemented, file changes, test results | Project managers |
| **INDEPENDENCE_REPORT.md** | Dependency removal, import strategy | DevOps/maintainers |
| **CHANGES.md** | Detailed changelog, backwards compatibility | Upgrading users |
| **VISUAL_OVERVIEW.md** | ASCII diagrams, visual explanations | Visual learners |
| **DOCUMENTATION_INDEX.md** | Navigation guide for all docs | New readers |
| **demonstration.py** | Runnable code examples | Learning-by-doing |
| **AGENTS.md** | This file — project overview for AI agents | AI assistants |

---

## 🔍 Key Code Locations

### Main Node Implementation
- **File:** `nodes.py` (606 lines)
- **Class:** `DwRestorator`
- **Method:** `dwrestore()` — Main entry point

### Core Algorithms
| Method | Lines | Purpose |
|--------|-------|---------|
| `_estimate_affine_transform()` | 45 | Compute transformation matrix |
| `_transform_point()` | 8 | Apply affine to point |
| `_restore_keypoints_relative()` | 80 | Restore body/hand keypoints |
| `_restore_face_keypoints()` | 60 | Restore face landmarks |
| `_zero_out_of_canvas()` | 35 | Clamp to canvas bounds |
| `_generate_pose_image()` | 45 | Render visualization |

### Pose Handling
- **File:** `pose_visualization.py`
- Functions: `decode_json_as_poses()`, `draw_poses()`

### Type Definitions
- **File:** `pose_types.py`
- Classes: `Keypoint`, `BodyResult`, `HandResult`, `FaceResult`, `PoseResult`

---

## 🐛 Recent Bug Fixes

### JSON Serialization Issue (Jan 2, 2026)
**Problem:** `SavePoseKpsAsJsonFile` node failed with `TypeError: Object of type float32 is not JSON serializable`

**Root Cause:** NumPy/torch arithmetic resulted in numpy scalar types incompatible with JSON

**Solution:**
1. Added `convert_to_python_types()` utility function
2. Wrapped all numeric calculations with `float()` during restoration
3. Applied recursive conversion before returning pose data
4. Created documentation: `JSON_SERIALIZATION_FIX.md`

---

## 📈 Performance Characteristics

### Computational Complexity
- **Time:** O(n) per keypoint type (body, hands, face independent)
- **Space:** O(n) for storing pose data and transforms
- **GPU Support:** Optional for tensor operations in visualization

### Canvas Sizes
- **Default:** 512×512 (body/hands)
- **Configurable:** Any dimensions supported
- **Example:** 768×1365 for taller poses

### Typical Execution Times
- Body restoration: ~5–10ms (18 keypoints)
- Hand restoration: ~2–5ms per hand (21 points)
- Face restoration: ~5–10ms (68–70 points)
- Image generation: ~50–100ms (depends on canvas size)
- **Total:** ~100–150ms per pose

---

## 🔗 Integration Points

### Upstream (Requires)
- **ComfyUI Core:** Node registration, POSE_KEYPOINT type
- **DWPose Detector:** Generates initial pose data
- **Image Input:** For reference frame loading

### Downstream (Used By)
- **SavePoseKpsAsJsonFile:** Exports pose data to JSON
- **Image Generation Nodes:** Consume pose visualization
- **ControlNet Nodes:** Take restored pose as input
- **Animation/Video Nodes:** Apply restored poses to sequences

### External Integrations
- **React Frontend:** Optional UI in ComfyUI sidebar
- **ComfyUI API:** REST endpoints for headless operation
- **Custom Scripts:** Direct Python import usage

---

## 🌍 Localization & i18n

### Supported Languages
- **English (en):** Default, complete
- **Chinese (zh):** Simplified, complete

### How to Add Languages
1. Create `ui/public/locales/{lang}/main.json`
2. Add translations for keys in English version
3. React-i18next auto-detects browser language
4. Falls back to English if unavailable

### Current Locales
```
ui/public/locales/
├── en/main.json         # English translations
└── zh/main.json         # Chinese (simplified)
```

---

## 🎓 Understanding the Code

### For New Contributors

**Start Here:**
1. Read `README.md` → Overview
2. Read `QUICK_REFERENCE.md` → Algorithm summary
3. Read `IMPLEMENTATION_GUIDE.md` → Detailed concepts
4. Review `demonstration.py` → Code examples
5. Explore `nodes.py` → Implementation details

**Key Concepts to Master:**
- How missing keypoints are detected (all zeros)
- Why affine transformation is needed (maintaining proportions)
- How skeleton hierarchy works (parent-child relationships)
- Why visualization needs clamping (out-of-canvas handling)

### For Code Review

**Critical Review Points:**
1. **Affine Transform Accuracy:** Check `_estimate_affine_transform()`
2. **Hierarchy Coverage:** Verify all BODY/HAND/FACE hierarchies
3. **Type Conversion:** Ensure no numpy types leak to output
4. **Canvas Bounds:** Check `_zero_out_of_canvas()` logic
5. **Confidence Handling:** Verify `reduce_confidence` parameter

---

## 🚨 Known Limitations & Future Work

### Current Limitations
1. **Face Restoration:** No fixed hierarchy, uses local neighbors (less reliable)
2. **GPU Optimization:** Only image generation can use GPU
3. **Batch Processing:** Processes one pose at a time
4. **Canvas Size:** Must be specified per pose, no auto-detection

### Potential Improvements
1. **Face Hierarchy:** Implement facial structure constraints
2. **Multi-Person:** Handle multiple people in single image
3. **Temporal Consistency:** Smooth keypoints across frames
4. **Advanced Affine:** Use RANSAC for outlier-resistant fitting
5. **Performance:** Vectorize restoration for batch inputs

---

## 📝 License & Attribution

**License:** GNU General Public License v3 (GPLv3)

**Author:** Tony Zachesov (NCRIMSK)  
**Repository:** https://github.com/NCRIMSK/dwpose_restorator

**Dependencies:**
- NumPy (BSD)
- OpenCV (Apache 2.0)
- PyTorch (BSD)
- React (MIT)
- Vite (MIT)

---

## 💡 Quick Tips for Agents/Developers

1. **Node Always Returns Tuple:** `(image_tensor, pose_dict)`
2. **JSON Serialization:** Use `convert_to_python_types()` before export
3. **Missing Keypoints:** Detected by `x == 0 and y == 0 and conf == 0`
4. **Canvas Bounds:** Important for visualization, not calculations
5. **Hierarchy Order:** Process parents before children (automatic in loops)
6. **Confidence:** Inherits from parent, optionally reduced
7. **UI Build:** Run `npm run build` in `ui/` before ComfyUI reload
8. **Fallback Import:** System gracefully handles missing dependencies

---

## 🔗 Related Resources

- **ComfyUI Docs:** https://docs.comfy.org/
- **DWPose:** https://github.com/RyumakAkira/DWPose/
- **OpenCV Affine:** https://docs.opencv.org/master/d2/d00/cv_2imgproc_2imgproc.hpp.src.html
- **PyTorch Tensors:** https://pytorch.org/docs/stable/tensors.html
- **React i18n:** https://react.i18next.com/

---

**Last Updated:** January 2, 2026  
**Version:** 0.0.1  
**Status:** Production-Ready
