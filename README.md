# 🖱️ AI Virtual Mouse

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/framework-MediaPipe-green.svg)](https://mediapipe.dev/)
[![PyAutoGUI](https://img.shields.io/badge/library-PyAutoGUI-orange.svg)](https://pyautogui.readthedocs.io/)

A state-of-the-art, hands-free virtual mouse that translates real-time hand gestures into precise system mouse controls. Built using **OpenCV**, **MediaPipe Hands**, and **PyAutoGUI**, it features highly optimized adaptive filtering algorithms (including the **One Euro Filter** and **Kalman Filter**) to achieve smooth, lag-free cursor movements and robust click actions without physical contact.

---

## ✨ Key Features

- **🎯 Zero-Contact Tracking**: Precise hand tracking using your standard webcam, mapping movements directly to your screen space.
- **⚡ Adaptive Speed-Adaptive Smoothing**: Uses the **One Euro Filter** to provide heavy jitter elimination when your hand is stationary, and virtually zero lag when moving quickly.
- **💎 Click-Drift Mitigation**: Configured to track the stable **Index Knuckle (MCP Joint)** for relative movement, keeping the cursor completely stationary during click/pinch actions to avoid accidental clicking.
- **🖥️ Threaded Camera Pipeline**: Decouples webcam frames capture from main-thread landmark detection and UI rendering to maximize frames per second (FPS).
- **🎨 Premium HUD Interface**: An overlay displaying system active status, real-time FPS, hand detection quality, interactive finger states card, and click event indicators.

---

## 🖐️ Gesture Control Directory

| Action | Gesture | Description | Visual Feedback |
| :--- | :--- | :--- | :--- |
| **Move Cursor** | **Move Hand** | Hover hand inside the tracking zone (relative movement follows the index knuckle). | Normal landmark points |
| **Left Click** | **Index Pinch** | Pinch index finger tip to thumb tip. | Midpoint glowing green |
| **Right Click** | **Ring Pinch** / **Double Tap** | Pinch ring finger tip to thumb tip, or double-tap index pinch. | Highlighted ring tip |
| **Double Click** | **Middle Pinch** | Pinch middle finger tip to thumb tip. | Highlighted middle tip |
| **Scroll Mode** | **Pinch + Vertical Slide** | Pinch index finger and thumb, then move your hand up or down. | Vertical scroll events |
| **Drag & Drop** | **Pinch & Hold** | Pinch index finger and thumb and hold for `0.35s` to drag; release pinch to drop. | Screen drag indicator |
| **System Pause** | **Closed Fist** | Fold all fingers into a fist to temporarily freeze cursor control. | Blinking orange status dot |
| **System Resume**| **Open Palm** | Extend all fingers to resume active tracking. | Steady green status dot |
| **Screenshot** | **Three Finger Pose** | Extend index, middle, and ring fingers while keeping thumb and pinky folded. | Screenshot saved message |

---

## 🛠️ Project Structure

```bash
├── app.py                  # Main entrypoint, handles frame capture, UI rendering & main loop.
├── gesture_detector.py     # Detects hand landmarks, checks handedness, and classifies gestures.
├── mouse_controller.py     # Maps camera coordinates to screen pixels and smooths mouse movement.
├── config.py               # Houses default configurations and utility properties.
├── config.json             # Persistent JSON configurations.
└── requirements.txt        # Third-party dependency list.
```

---

## ⚙️ Installation & Setup

### Prerequisites

Ensure you have Python 3.10 or newer installed.

### 1. Install Dependencies

Clone the repository to your local machine, open your terminal in the root directory, and run:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Launch the virtual mouse application by running:

```bash
python app.py
```

Move your hand in front of the camera. To exit, press `Q` while focused on the camera preview window, or press `Ctrl + C` in your terminal.

---

## 🔧 Configuration & Parameter Tuning

All controls can be tuned dynamically by modifying the values inside `config.json`. Key parameters include:

### Cursor Sensitivity & Movement
- `SENSITIVITY` (`1.6`): Overall speed factor of the cursor.
- `ACCELERATION_FACTOR` (`2.0`): Multiplier that accelerates the cursor based on hand movement speed.
- `DEADZONE_PIXELS` (`3`): Ignore coordinate displacement below this threshold to completely eliminate cursor shake at rest.
- `TRACKING_LANDMARK_ID` (`5`): The joint index to track (`5` corresponds to the Index Finger Knuckle / MCP, which is highly stable). Use `"MIDPOINT"` to track the index-thumb midpoint instead.

### Smoothing Filters
- `SMOOTHING_METHOD` (`"one_euro"`): Choose between `"one_euro"`, `"kalman"`, and `"ema"`.
- `ONE_EURO_FC_MIN` (`0.1`): Minimum cutoff frequency for One Euro Filter. Lowering this values increases smoothing when slow (ideal for aiming at small UI elements).
- `ONE_EURO_BETA` (`0.15`): Speed coefficient. Higher values minimize lag when moving fast.

### Gesture Timings
- `CLICK_COOLDOWN_SEC` (`0.2`): Delay between successive click operations.
- `DRAG_START_DELAY` (`0.35`): Time in seconds required to hold a pinch to engage dragging.

---

## 📜 License

This project is open-source and available under the MIT License.
