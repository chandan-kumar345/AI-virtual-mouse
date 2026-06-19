import os
import json
import pyautogui
from typing import Dict, Any

# Get screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Default settings (used if config.json does not exist)
DEFAULTS: Dict[str, Any] = {
    "CAMERA_INDEX": 0,
    "CAM_WIDTH": 640,
    "CAM_HEIGHT": 480,
    "FRAME_RATE": 30,
    
    # Cursor Control Settings
    "SENSITIVITY": 1.6,
    "ACCELERATION_FACTOR": 2.0,  # Speed-based acceleration multiplier
    "DEADZONE_PIXELS": 3,        # Tiny movement threshold to ignore jitter
    "TRACKING_LANDMARK_ID": 5,   # Stable landmark: 5 (Index MCP), 8 (Index Tip), or "MIDPOINT"
    
    # Smoothing Settings
    "SMOOTHING_METHOD": "one_euro",  # Default to One Euro Filter
    "SMOOTHING_ALPHA": 0.15,         # For EMA
    "LANDMARK_EMA_MIN_ALPHA": 0.08,  # Minimum alpha for adaptive landmark EMA
    "LANDMARK_EMA_MAX_ALPHA": 0.95,  # Maximum alpha for adaptive landmark EMA
    
    # One Euro Filter parameters
    "ONE_EURO_FC_MIN": 0.1,          # Min cutoff frequency (smooths when slow)
    "ONE_EURO_BETA": 0.15,           # Speed coefficient (reduces lag when fast)
    "ONE_EURO_FC_D": 1.0,            # Cutoff frequency for velocity
    
    # Kalman Filter parameters
    "KALMAN_Q_ACCEL": 0.1,           # Process noise (acceleration variance)
    "KALMAN_R_MEAS": 0.5,            # Measurement noise (measurement variance)
    
    # Active Tracking Box (normalized coordinates [0.0 - 1.0]) - updated to full frame
    "TRACKING_BOX_LEFT": 0.0,
    "TRACKING_BOX_RIGHT": 1.0,
    "TRACKING_BOX_TOP": 0.0,
    "TRACKING_BOX_BOTTOM": 1.0,
    
    # Gesture Thresholds (normalized relative to hand size)
    "PINCH_THRESHOLD": 0.13,               # Threshold to engage Left Click / Drag
    "PINCH_RELEASE_THRESHOLD": 0.16,       # Hysteresis threshold to release Left Click / Drag
    "RIGHT_CLICK_THRESHOLD": 0.13,          # Threshold to engage Right Click
    "RIGHT_CLICK_RELEASE_THRESHOLD": 0.16,  # Hysteresis threshold to release Right Click
    "DOUBLE_CLICK_THRESHOLD": 0.10,         # Threshold to engage Double Click
    "DOUBLE_CLICK_RELEASE_THRESHOLD": 0.12, # Hysteresis threshold to release Double Click
    
    # New Touch and Double Tap Gestures
    "TOUCH_THRESHOLD": 0.15,                # Normalized distance threshold for touch detection
    "RELEASE_THRESHOLD": 0.20,              # Normalized distance threshold for release detection
    "GESTURE_CONFIRM_FRAMES": 2,            # Number of frames to confirm touch or release
    "RAPID_MOVEMENT_THRESHOLD": 1.5,        # Normal velocity limit to block accidental click triggers
    "TAP_MAX_DURATION": 0.25,               # Max duration of a tap gesture in seconds
    "DOUBLE_TAP_MIN_WINDOW": 0.10,          # Min interval between consecutive taps
    "DOUBLE_TAP_MAX_WINDOW": 0.30,          # Max interval between consecutive taps
    "LATENCY_COMPENSATION_SEC": 0.06,       # Motion prediction time step for Kalman Filter
    "STATIONARY_VELOCITY_THRESHOLD": 15.0,  # Velocity threshold to zero out movement drift
    "DRAG_START_DELAY": 0.35,               # Hold time in seconds to initiate Drag & Drop
    
    # Scroll Settings
    "SCROLL_SENSITIVITY": 15,
    "SCROLL_DEADZONE": 5,
    "SCROLL_DECAY": 0.8,                     # Decay factor for scroll inertia (friction)
    "SCROLL_START_THRESHOLD": 0.04,          # Vertical finger movement to engage scroll
    
    # Gesture Debouncing & Cooldowns
    "CLICK_DEBOUNCE_FRAMES": 10,
    "CLICK_COOLDOWN_SEC": 0.20,              # Minimum seconds between click triggers (cooldown 150-250ms)
    "GESTURE_DEBOUNCE_FRAMES": 3,            # Sliding window frames for gesture stability
    "PAUSE_GESTURE_ENABLED": True,
    
    # Hand Restrictions
    "RIGHT_HAND_ONLY": False,                 # Restrict virtual mouse control to the user's right hand only
    "MIRROR_HAND_LABEL": True,               # Flip hand label check due to horizontal mirroring
}

# Dynamically populate the module globals with defaults
for key, val in DEFAULTS.items():
    globals()[key] = val

CONFIG_FILE = "config.json"

def load_config() -> None:
    """Loads configuration from JSON file, dynamically updating module globals."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            for key in DEFAULTS:
                if key in data:
                    globals()[key] = data[key]
            print(f"[Config] Loaded configuration from '{CONFIG_FILE}'.")
        except Exception as e:
            print(f"[Config] Error loading configuration: {e}. Using defaults.")
    else:
        save_config()

def save_config() -> None:
    """Saves the current configuration globals to config.json."""
    try:
        data = {key: globals()[key] for key in DEFAULTS}
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[Config] Saved template configuration to '{CONFIG_FILE}'.")
    except Exception as e:
        print(f"[Config] Error saving configuration: {e}")

# Load configuration at import time
load_config()

