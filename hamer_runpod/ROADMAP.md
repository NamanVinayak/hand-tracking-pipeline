# Egocentric Hand Tracking Pipeline - Strategic Roadmap

**Last Updated:** January 1, 2026
**Purpose:** Production roadmap for building competitive egocentric data pipeline for robotics AI training

---

## 🎯 Market Context

### What Robotics Companies Need (Priority Order)

1. ⭐⭐⭐⭐⭐ **Hand-Object Interaction** (CRITICAL)
   - What objects are being manipulated
   - How hands grip objects (contact points)
   - 6D object pose (position + rotation in 3D)
   - Object segmentation masks

2. ⭐⭐⭐⭐ **Full Body Pose**
   - Arm positions and orientations
   - Torso pose
   - Whole-body coordination

3. ⭐⭐⭐ **Scene Context**
   - Objects in environment
   - Spatial relationships
   - Scene understanding

4. ⭐⭐⭐ **Task Labels**
   - Action annotations ("opening door", "pouring water")
   - Activity recognition

### Competitive Landscape

**Apple EgoDex** (Free, May 2025 release)
- 829 hours of data
- Hand pose + body pose + object 6D poses + scene understanding
- Vision Pro capture (multi-camera, depth sensors)
- Sets quality baseline for market

**HOT3D** (Meta, CVPR 2025)
- Hand pose + object 6D poses + object meshes
- Focus: hand-object interaction
- Quest 3 + Aria glasses capture

**Ego-Exo4D** (Meta, 2025)
- 1,286 hours from 740 participants
- Hand + body + object segmentation + activity labels
- GoPro-based (similar to our setup)

**Our Advantage:**
- India cost arbitrage (1/5 US costs)
- Professional Western-style environments
- Faster iteration than academic datasets

**Our Gap:**
- Missing object tracking (competitors have this)
- Missing scene understanding
- Missing activity labels

---

## 📊 Current State (Phase 0 - COMPLETED)

### ✅ What We Have

**Hand Tracking Pipeline (Production Ready)**
- HaMeR-based 3D hand pose estimation
- 21 MANO joints per hand with 3D coordinates
- ViTPose 2D keypoint overlay (pixel-perfect)
- RunPod serverless GPU deployment
- Cloudflare R2 storage integration
- Apple EgoDex visual styling (gradient joints, minimal aesthetic)

**Recent Addition (Dec 31, 2025):**
- 2D arm skeleton overlay (shoulder → elbow → wrist)
- Extracted from ViTPose (indices 5-10)
- Blue skeleton for left arm, orange for right
- Works in video overlay, NOT yet in 3D viewer

**Technical Stack:**
- HaMeR (hand mesh recovery)
- ViTPose-H (133 body keypoints)
- ViTDet (person detection)
- Docker multi-layer strategy (base + handler)
- Rerun.io for 3D visualization

**Output Format:**
```json
{
  "frames": [
    {
      "frame_idx": 0,
      "hands": [
        {
          "side": "left",
          "joints_3d": [[x,y,z], ...],  // 21 hand joints
          "camera_t": [tx, ty, tz],
          "vitpose_2d": [[x,y], ...],   // 2D keypoints
          "arm_keypoints_2d": {         // NEW
            "shoulder": [x, y, conf],
            "elbow": [x, y, conf],
            "wrist": [x, y, conf]
          }
        }
      ]
    }
  ]
}
```

**Processing Cost:** ~$0.10-0.15 per minute of video (30 FPS)

### ❌ What We're Missing (vs Competitors)

1. **Object Tracking** - CRITICAL GAP
   - No object segmentation
   - No object identification
   - No 6D object poses
   - No hand-object contact points

2. **3D Arm Visualization**
   - Arms only show in 2D overlay
   - Missing from 3D Rerun viewer
   - Only have 2D pixel coordinates, not 3D positions

3. **Scene Understanding**
   - No depth estimation
   - No scene segmentation
   - No spatial context

4. **Activity Labels**
   - No task annotations
   - No action recognition

---

## 🚀 Phase 1: 3D Arm Visualization (IN PROGRESS)

**Status:** Planning
**Timeline:** 30 minutes - 2 days (depending on approach)
**Priority:** Medium (nice-to-have for demos)

### Problem
ViTPose gives us 2D arm keypoints (pixels) but no depth (z-coordinate).
Need to estimate 3D positions to render arms in 3D viewer alongside hands.

### Three Approaches

#### Option 1A: Simple Depth Estimation ⭐ (RECOMMENDED FOR NOW)
**Timeline:** 30 minutes
**Cost:** $0
**Accuracy:** 70-80%

**Method:**
1. Use hand wrist 3D position from HaMeR as depth reference
2. Project 2D arm keypoints to 3D using camera parameters
3. Place elbow/shoulder at same depth as wrist
4. Render in Rerun viewer

**Pros:**
- ✅ Fast implementation
- ✅ Uses existing data
- ✅ Good enough for demos

**Cons:**
- ⚠️ Assumes arm is flat (no depth variation)
- ⚠️ Not anatomically perfect

**When to use:** Immediate demos, customer validation

---

#### Option 1B: Biomechanical Arm Model
**Timeline:** 2-3 hours
**Cost:** $0
**Accuracy:** 85-90%

**Method:**
1. Use human anatomy constraints (arm length ratios)
2. Solve inverse kinematics for elbow/shoulder 3D position
3. Satisfy constraints:
   - Upper arm length: ~35cm
   - Forearm length: ~28cm
   - 2D projection matches ViTPose keypoints

**Pros:**
- ✅ More anatomically correct
- ✅ Better depth estimation
- ✅ Still uses existing data

**Cons:**
- ⚠️ More complex math
- ⚠️ Arm lengths vary per person

**When to use:** If customers complain Option 1A looks "off"

---

#### Option 1C: Full 3D Pose Model (WHAM)
**Timeline:** 1-2 days
**Cost:** +30% GPU processing (+$0.03-0.05 per minute)
**Accuracy:** 95%+

**Method:**
1. Deploy WHAM (Reconstructing World-grounded Humans with Accurate Motion)
2. Designed specifically for egocentric video
3. Outputs SMPL body mesh (full skeleton + surface)
4. Extract shoulder/elbow/wrist 3D positions
5. Combine with HaMeR hand data

**Pros:**
- ✅ State-of-the-art accuracy (CVPR 2024)
- ✅ True 3D estimation (not projection)
- ✅ Full body mesh (torso, legs available if needed)
- ✅ Future-proof for whole-body robot control

**Cons:**
- ❌ GPU memory: +3-4 GB (may need larger instances)
- ❌ Processing time: +30-40%
- ❌ Higher cost per video
- ❌ Complex integration (2 days work)

**When to use:**
- Production deliverables (not just demos)
- Customers need whole-body robot control
- After validating market demand

---

### Recommendation for Phase 1

**Do Option 1A immediately** (30 mins) to:
- Get 3D arms working for demos
- Show customers and get feedback
- Validate if anyone cares about 3D arm accuracy

**Then decide:**
- If customers say "looks good!" → keep it, move to Phase 2
- If customers say "arms look slightly off" → upgrade to Option 1B
- If customers demand "production quality" → deploy Option 1C (WHAM)

**DO NOT spend 2 days on WHAM until we validate demand.**

---

## 🎯 Phase 2: Object Tracking (HIGHEST PRIORITY)

**Status:** Not started
**Timeline:** 3-4 days
**Priority:** ⭐⭐⭐⭐⭐ CRITICAL
**Why:** This is the biggest competitive gap vs Apple EgoDex / HOT3D

### Problem
Robotics companies need to know:
- What objects are being manipulated
- Where objects are in 3D space
- How hands interact with objects

Currently we only track hands, not objects.

### Approach: SAM2 + Object Identification

**Phase 2A: Object Segmentation (2 days)**

**Tool:** SAM2 (Meta's Segment Anything Model 2)
- State-of-the-art video segmentation (2024)
- Can track any object through video
- Works with manual prompts or auto-detection

**Method:**
1. User clicks object in first frame (or auto-detect from hand bounding box)
2. SAM2 propagates segmentation mask through entire video
3. Output: pixel-level object masks per frame
4. Save masks + bounding boxes in JSON

**Output format:**
```json
{
  "frame_idx": 0,
  "hands": [...],
  "objects": [
    {
      "object_id": "obj_001",
      "mask": [...],  // pixel mask
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}
```

**GPU Cost:** +30-40% processing time
**Storage:** Masks are large (~5-10x JSON size)

---

**Phase 2B: Object Identification (1 day)**

**Tool:** CLIP (OpenAI) or OWL-ViT
- Zero-shot object classification
- Identify object types from images

**Method:**
1. Extract object crop from segmentation mask
2. Run CLIP to classify object type
3. Add label to JSON: "cup", "bottle", "laptop", etc.

**Output format:**
```json
{
  "objects": [
    {
      "object_id": "obj_001",
      "label": "coffee cup",
      "confidence": 0.92,
      "mask": [...],
      "bbox": [...]
    }
  ]
}
```

---

**Phase 2C: 6D Object Pose (OPTIONAL - 2-3 days)**

**Tool:** FoundationPose or MegaPose
- Estimate 6D pose (3D position + 3D rotation)
- Requires object CAD models or learned priors

**Output format:**
```json
{
  "objects": [
    {
      "object_id": "obj_001",
      "label": "coffee cup",
      "pose_6d": {
        "position": [x, y, z],
        "rotation": [qw, qx, qy, qz]  // quaternion
      }
    }
  ]
}
```

**Complexity:** High (only do if customers specifically request)

---

### Deliverable After Phase 2

**JSON Output:**
- Hand tracking (21 joints per hand)
- Object segmentation masks
- Object labels
- Object bounding boxes
- (Optional) 6D object poses

**Visualization:**
- 2D video: hands + objects highlighted
- 3D viewer: hands + object positions

**Value Proposition:**
- ✅ Matches HOT3D capabilities
- ✅ Hand-object interaction data
- ✅ Competitive with Apple EgoDex (minus scene understanding)

**Pricing Impact:**
- Can charge 2-3x current rates
- Justify higher prices vs basic hand tracking

---

## 🦾 Phase 3: Full 3D Body Pose (WHAM Integration)

**Status:** Not started
**Timeline:** 1-2 days
**Priority:** Medium (do after Phase 2)
**Depends on:** Phase 1 completion

### Why Do This

**After Phase 2 (object tracking), add full body for:**
- Whole-body robot control (humanoid robots)
- Torso orientation context
- Better arm accuracy than simple estimation
- Premium tier offering

### Implementation

**Model:** WHAM (CVPR 2024)
- Designed for egocentric video
- Outputs SMPL body mesh
- Includes: head, torso, arms, legs

**Integration Points:**
1. Add WHAM to Docker base image
2. Run WHAM before HaMeR in pipeline
3. Extract body keypoints from SMPL mesh
4. Combine with HaMeR hand data
5. Render full skeleton in Rerun

**Output format:**
```json
{
  "frame_idx": 0,
  "body_pose": {
    "smpl_params": [...],
    "joints_3d": [...],  // Full body skeleton (17-24 joints)
    "mesh_vertices": [...]  // Optional: full body surface
  },
  "hands": [...],
  "objects": [...]
}
```

**GPU Memory:** +3-4 GB
**Processing Time:** +30-40%
**Cost:** +$0.03-0.05 per minute

### Deliverable After Phase 3

**Complete Dataset:**
- ✅ Hand tracking (21 joints)
- ✅ Full body pose (SMPL mesh)
- ✅ Object tracking
- ✅ Object labels

**Market Position:**
- Matches or exceeds Apple EgoDex for egocentric data
- Competitive with Ego-Exo4D
- Only missing: scene understanding, depth maps

---

## 🌍 Phase 4: Scene Understanding (ADVANCED)

**Status:** Future work
**Timeline:** 1 week
**Priority:** Low (only if customers demand)

### Components

**4A: Depth Estimation**
- Tool: MiDaS or DepthAnything
- Output: Per-pixel depth maps
- Use case: 3D scene reconstruction

**4B: Scene Segmentation**
- Tool: Mask2Former or OneFormer
- Output: Semantic scene masks (floor, wall, furniture)
- Use case: Scene understanding

**4C: 3D Scene Reconstruction**
- Tool: DROID-SLAM or NeuralRecon
- Output: 3D mesh of environment
- Use case: Robot navigation

### When to Implement

**Only if customers specifically request:**
- Robot navigation in environment
- Spatial reasoning tasks
- 3D scene understanding

**Otherwise:** Focus on hand-object interaction (Phase 2-3)

---

## 📈 Pricing Strategy by Phase

### Current Offering (Phase 0)
**Product:** Hand tracking only
**Price:** $X per hour of video
**Justification:** Basic egocentric data

### After Phase 1 (3D Arms)
**Product:** Hand + arm tracking
**Price:** $X per hour (no increase)
**Justification:** Quality improvement, not new feature

### After Phase 2 (Object Tracking)
**Product:** Hand + object interaction
**Price:** $2-3X per hour
**Justification:** Matches HOT3D capabilities, critical for robotics

### After Phase 3 (Full Body)
**Product:** Full body + hands + objects
**Price:** $4-5X per hour (premium tier)
**Justification:** Complete dataset, matches Apple EgoDex

### After Phase 4 (Scene Understanding)
**Product:** Complete scene + body + objects
**Price:** $6-8X per hour (enterprise tier)
**Justification:** Exceeds all competitors except full research datasets

---

## 🎯 Recommended Execution Order

### Week 1 (Jan 1-7, 2026)
- ✅ Complete Phase 1 Option 1A (3D arm estimation - 30 mins)
- ✅ Test with customers, get feedback
- 🚀 **START Phase 2A** (SAM2 object segmentation - 2 days)

### Week 2 (Jan 8-14, 2026)
- 🚀 Complete Phase 2A (object masks)
- 🚀 Implement Phase 2B (object identification - 1 day)
- 🚀 Test object tracking on sample videos
- 📊 Show customers, validate pricing

### Week 3 (Jan 15-21, 2026)
- **Decision point:** Do customers want 6D poses? (Phase 2C)
  - If yes → implement (2-3 days)
  - If no → skip, move to Phase 3
- Start Phase 3 planning (WHAM integration)

### Week 4 (Jan 22-28, 2026)
- 🚀 Implement Phase 3 (WHAM full body - 1-2 days)
- 🚀 Test complete pipeline
- 📊 Prepare customer demos with full feature set

### Month 2+ (Feb 2026)
- Scale up data collection with Phase 2+3 features
- Monitor customer feedback
- Consider Phase 4 only if demanded

---

## ⚠️ Critical Success Factors

### 1. Customer Validation at Each Phase
- Don't build Phase 3 until Phase 2 is validated
- Don't build Phase 4 until customers explicitly request it
- **Avoid over-engineering**

### 2. Cost Management
- Each phase adds GPU cost
- Ensure pricing covers increased costs + margin
- Monitor RunPod spending

### 3. Quality over Speed
- Apple EgoDex is free (May 2025)
- Must differentiate on:
  - ✅ India cost advantage
  - ✅ Western-style environments
  - ✅ Professional quality
  - ✅ Faster iteration

### 4. Focus on Phase 2 (Objects)
- **This is the biggest gap vs competitors**
- Hand-object interaction is critical for robotics
- Highest ROI for development time

---

## 🔧 Technical Debt to Address

### Before Scaling
1. Document RunPod deployment process
2. Automate worker management (avoid GPU memory issues)
3. Create batch processing scripts for multiple videos
4. Set up monitoring/alerting for failures
5. Build customer portal for video upload + download

### Before Phase 2
1. Optimize storage costs (masks are large)
2. Consider compression strategies
3. Build visualization tools for object masks
4. Update CLAUDE.md with object tracking documentation

### Before Phase 3
1. Test GPU memory limits with WHAM + HaMeR + SAM2
2. May need to upgrade RunPod instance type
3. Optimize processing pipeline (parallel vs sequential)

---

## 📚 Key Resources

### Models to Evaluate
- **WHAM:** https://github.com/yohanshin/WHAM
- **SAM2:** https://github.com/facebookresearch/segment-anything-2
- **FoundationPose:** https://github.com/NVlabs/FoundationPose
- **CLIP:** https://github.com/openai/CLIP

### Competitor Datasets
- **Apple EgoDex:** https://github.com/apple/ml-egodex (May 2025)
- **HOT3D:** https://facebookresearch.github.io/hot3d/
- **Ego-Exo4D:** https://ego-exo4d-data.org/

### Research Papers
- WHAM: CVPR 2024
- SAM2: arXiv 2024
- HaMeR: CVPR 2023
- ViTPose: NeurIPS 2022

---

## 🎬 Next Steps (Immediate)

1. **TODAY:** Implement Phase 1 Option 1A (30 mins)
2. **THIS WEEK:** Start Phase 2A (SAM2 integration)
3. **PAUSE:** Wait for customer feedback before Phase 3

**Question to answer before proceeding:**
- Do customers care more about 3D arm accuracy or object tracking?
- **Current hypothesis:** Object tracking is more valuable

**Decision criteria:**
- If customers request 6D poses → prioritize Phase 2C
- If customers request full body → prioritize Phase 3
- Otherwise → focus on Phase 2A+2B only

---

**Document Status:** Living document, update after each phase completion
**Owner:** Naman
**Last Review:** January 1, 2026
