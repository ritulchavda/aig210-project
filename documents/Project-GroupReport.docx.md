# Project - Final Group Report

## 1. Project Title

Vision-Based Game Controller Using Hand Gesture Recognition (Vision-Controlled Tetris)

## 2. Problem Definition

Traditional keyboard and mouse interaction is effective but not always intuitive or accessible for natural, touchless control scenarios. Our project addresses this by building a real-time, vision-based game controller that allows players to control a 2D Tetris game using only hand gestures captured by a webcam. This problem is useful because it combines accessibility, human-computer interaction, and computer vision in a practical setting. A robust gesture interface can support users who prefer hands-free interaction, reduce dependency on physical input devices, and demonstrate how lightweight AI-based interaction can be integrated into browser applications.

## 3. Dataset

### Source

Our dataset is custom-collected in real time using a webcam and MediaPipe Hand Landmarker.

### Size and Content

The static gesture dataset used for profile training contains:

- Palm: 160 samples
- Index: 160 samples
- TwoFinger: 160 samples

Total static samples: 480 hand landmark samples.

Each sample is represented as 21 hand landmarks projected to 2D (x, y), then converted into a normalized feature vector of length 42.

Motion gestures (Left flick, Right flick, Down flick) are detected online from short temporal wrist trajectories and therefore are not stored as a separate offline file-based dataset.

### Sample Images/Snapshots

Please paste screenshots from your run into this report before submission:

- Camera panel with hand wireframe and recognized gesture text.
- Motion arrow overlay examples (Left, Right, Down).
- Game panel during normal gameplay.
- Game panel showing pause or game-over state.

## 4. Ground Truth

Ground truth labels for static gestures were created manually during data collection in the training script.

- Label 1: Palm
- Label 2: Index
- Label 3: TwoFinger

Collection method:

- The operator presses `1`, `2`, or `3` to select the target class.
- While holding the chosen gesture, the system captures normalized landmark features.
- Collection continues until 160 samples are recorded for each class.

This process produces class-labeled feature vectors used to compute per-class centroids and matching thresholds.

## 5. Dataset Splitting, Preparation, and Preprocessing

### Splitting Strategy

We did not use a traditional train/validation/test split because the deployed classifier is profile-based (centroid + distance threshold), not a multi-epoch neural network. Instead, all collected samples were used to build user-specific gesture profiles for real-time deployment.

### Preprocessing

For each frame with a detected hand:

- Extract 21 landmarks from MediaPipe.
- Use the wrist landmark as origin and convert all points to relative coordinates.
- Normalize by the maximum landmark distance from the wrist to reduce scale effects.
- Flatten to a 42-dimensional vector.

This preprocessing provides translation and scale normalization and improves consistency under different hand positions and camera distances.

### Thresholding

Per-class threshold is computed from the 92nd percentile of intra-class distances to centroid plus a margin (0.02 in trainer). In the web app, an adaptive runtime threshold multiplier (1.5x) is used to improve robustness.

### Augmentation

No synthetic image augmentation was applied. Instead, robustness is improved through live capture variability (small natural pose, orientation, and lighting variation during manual sample collection).

## 6. Previous Work

Our project builds on prior work in hand-gesture recognition and gesture-based interaction systems, especially research that uses vision models for touchless control, HCI, and game-like interfaces. We adopted MediaPipe Hands/Hand Landmarker as our starting technical foundation for fast and reliable landmark extraction, then implemented a custom gesture layer and direct browser game integration. Compared with many previous works that focus on isolated recognition demos, our implementation emphasizes end-to-end usability: gesture capture, classification, control mapping, user feedback overlays, and full interactive gameplay in one system.

## 7. Method and Contributions

Our method combines static and motion gesture recognition into a browser-based game control loop. Static gestures (Palm, Index, TwoFinger) are recognized either by learned profile matching or a fallback rule-based detector. In the learned path, the app compares each normalized feature vector against stored gesture centroids and thresholds from `gesture_profiles.json`. If no reliable match is found, the system falls back to geometric finger-state rules (extended vs folded fingers). This hybrid strategy increases reliability and keeps the system functional even without a trained profile file.

Motion gestures are recognized using short-term temporal tracking of wrist movement. The app stores a recent history window (about 360 ms) and computes directional displacement. Dominant downward displacement triggers `Down` for soft drop, while horizontal wrist tracking is handled continuously through delta accumulation, allowing smoother lateral movement than one-shot event triggers. In the current implementation, `Left`/`Right` motion labels are displayed and logged, while actual left-right piece movement is primarily controlled by continuous wrist tracking.

Our main contributions are:

- A hybrid static-gesture pipeline (learned profiles + fallback rules).
- Real-time motion gesture recognition integrated with cooldown control to reduce repeated triggers.
- Handedness-aware command gating (`Right` hand required for gameplay actions).
- Complete integration into a fully playable Tetris engine with score, lines, level progression, and game-state feedback.
- A practical, reproducible training script that generates personalized gesture profiles.

These steps were chosen to balance speed, implementation complexity, and user experience in a course-scale project.

Implemented gameplay gesture mapping in the web app:

- `Index` -> Hard drop
- `TwoFinger` -> Rotate piece
- `Down` (motion) -> Soft drop
- Horizontal wrist movement -> Continuous move left/right

Note: `Palm` can be recognized by the vision pipeline but is not currently mapped to pause/resume in gameplay logic.

## 8. Outcome and Reflection

The method achieved the expected outcome: users can play Tetris in real time using hand gestures with webcam input. The system demonstrates stable detection under normal indoor lighting and reasonable camera framing. The most important success factor is combining multiple safeguards (profile matching, fallback rules, cooldowns, and motion windows), which makes control behavior more predictable. Key limitations include sensitivity to poor lighting/background clutter and occasional ambiguity between similar finger poses. Future improvements include multi-user testing, more gesture classes, confusion-matrix-based analysis, and adaptive per-user recalibration inside the web UI.

## 9. Evaluation

### Quantitative Evaluation

Configuration and measurable outputs from the implemented system:

- Static gesture classes trained: 3 (Palm, Index, TwoFinger)
- Motion gesture classes detected online: 3 (Left, Right, Down)
- Samples per static class: 160
- Total static samples: 480
- Feature length: 42
- Game board: 10 columns x 20 rows
- Rendered block size: 30 px
- Base drop interval: 780 ms, reduced by level down to a floor of 140 ms
- Learned profile file indicates all 3 classes trained successfully.
- Runtime hand detection confidence settings:
	- min_hand_detection_confidence = 0.55
	- min_hand_presence_confidence = 0.50
	- min_tracking_confidence = 0.50

Recorded profile thresholds from the current trained model:

- Palm threshold: 1.0667
- Index threshold: 1.2479
- TwoFinger threshold: 1.1223

Please insert your final demo metrics (e.g., average recognition accuracy per class, command latency, and session success rate) if you collected them during presentation rehearsal.

### Qualitative Evaluation

Qualitatively, the system provides clear user feedback through:

- Real-time hand wireframe visualization.
- On-screen recognized gesture text.
- Directional arrow overlays for motion gestures.
- Immediate in-game response for rotate, hard drop, soft drop, and horizontal movement.

Please paste representative screenshots for each gesture-action mapping and one short gameplay sequence snapshot series.

### Comparison with Previous Work

Compared with prior research-focused prototypes, this project prioritizes a deployable, browser-based end-to-end demo rather than only reporting recognition on static datasets. Our contribution is practical integration and real-time interaction quality, though we currently provide fewer benchmark-style statistical comparisons than large-scale academic studies.

## 10. Code Submission

Project code root: `d:/AIG210`

Main files:

- `index.html` - UI structure
- `styles.css` - UI styling
- `app.js` - gesture recognition + Tetris logic
- `train_gestures.py` - static gesture profile trainer
- `gesture_profiles.json` - trained gesture profile output
- `requirements-gestures.txt` - Python dependencies for trainer

### Setup and Run Instructions

1. Run the web app (required for gameplay):

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000` and allow webcam permission.

2. Optional: retrain static gesture profiles:

```powershell
pip install -r requirements-gestures.txt
python train_gestures.py --samples-per-gesture 160
```

3. Training controls:

- Press `1` for Palm, `2` for Index, `3` for TwoFinger.
- Press `S` to save once all classes reach target samples.

## 11. References

[1] S. Singhvi, N. Gupta, and S. M. Satapathy, "Virtual Gaming Using Gesture Recognition Model," in *Advances in Distributed Computing and Machine Learning*, Lecture Notes in Networks and Systems, Singapore: Springer, 2022, pp. 114-124. doi: 10.1007/978-981-16-4807-6_12.

[2] R. Husna, K. C. Brata, I. T. Anggraini, N. Funabiki, A. A. Rahmadani, and C.-P. Fan, "An Investigation of Hand Gestures for Controlling Video Games in a Rehabilitation Exergame System," *Computers*, vol. 14, no. 1, art. no. 25, Jan. 2025. doi: 10.3390/computers14010025.

[3] A. Paramarthalingam, A. Janarthanan, P. Arivunambi, S. S. Ariyangavu, H. Senthamaraikannan, and R. Ganapathy, "AI-Powered Virtual Mouse Control Through Hand Gestures With Computer Vision," in *Harnessing AI in Geospatial Technology for Environmental Monitoring and Management*, Advances in Geospatial Technologies, IGI Global, 2024, pp. 61-76. doi: 10.4018/979-8-3693-8104-5.ch003.

[4] S. I. Hafiz and W. K. Ming, "Real-Time Algorithms for Gesture-Based Control in Robotics and Gaming Systems," *PatternIQ Mining*, vol. 1, no. 4, Nov. 2024. doi: 10.70023/sahd/241102.

[5] Y. Zhu and B. Yuan, "Real-time hand gesture recognition with Kinect for playing racing video games," in *2014 International Joint Conference on Neural Networks (IJCNN)*, Beijing, China, Jul. 2014, pp. 3240-3246. doi: 10.1109/IJCNN.2014.6889481.

[6] Google AI Edge, "Hand Landmarker," MediaPipe Solutions Guide. [Online]. Available: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker. [Accessed: Apr. 5, 2026].

## 12. Showcase Link

Link: [Add your GitHub repository or hosted demo URL here]

## Submission and Group Work Declaration

We, Ritul Sunil Chavda, Omotoyosi Odele, and Shrunga Kundranda Ganapathi, declare that the attached project is entirely our own work and has been completed in accordance with the Seneca Academic Policy. We have not copied or reproduced any part of this assignment, either manually or electronically, from any unauthorized source, including but not limited to AI tools, homework-sharing websites, or other students' work, unless explicitly cited as references. We have not shared our work with others, nor have we received unauthorized assistance in completing this project.

Specify what each member has done towards the completion of this project:

| # | Name | Task(s) |
| :-- | :-- | :-- |
| 1 | Ritul Sunil Chavda | Implemented gesture classification logic, static profile matching, fallback gesture rules, and part of integration/debugging. |
| 2 | Omotoyosi Odele | Set up environment and hand landmark acquisition pipeline using MediaPipe; supported data collection and testing. |
| 3 | Shrunga Kundranda Ganapathi | Implemented game integration and control mapping; contributed to UI/gameplay flow and final system testing. |

