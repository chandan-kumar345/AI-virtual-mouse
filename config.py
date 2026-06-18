import pyautogui

# Get screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera Settings
CAMERA_INDEX = 0          # Default webcam
CAM_WIDTH = 640           # Camera capture width
CAM_HEIGHT = 480          # Camera capture height
FRAME_RATE = 30           # Targeted frame rate

# Cursor Control Settings
# Sensitivity: Higher value means the cursor moves faster relative to hand movements
SENSITIVITY = 1.5

# Smoothing factor (Exponential Moving Average): Range [0.0, 1.0]
# Lower values mean smoother movement but slight lag; higher means responsive but jittery
SMOOTHING_ALPHA = 0.25

# Active Tracking Box (normalized coordinates relative to camera frame size [0.0 - 1.0])
# This creates a smaller region in the camera frame mapped to the entire screen.
TRACKING_BOX_LEFT = 0.2    # 20% from left
TRACKING_BOX_RIGHT = 0.8   # 80% from left (width = 60%)
TRACKING_BOX_TOP = 0.2     # 20% from top
TRACKING_BOX_BOTTOM = 0.7  # 70% from top (height = 50%)

# Distance Thresholds for Gestures (normalized relative to image dimensions or hand size)
PINCH_THRESHOLD = 0.035          # Distance below which index & thumb is a pinch (Left click)
RIGHT_CLICK_THRESHOLD = 0.035    # Distance below which middle & thumb is a pinch
DOUBLE_CLICK_THRESHOLD = 0.03    # Distance below which index & middle are together
SCROLL_SENSITIVITY = 15          # Scroll speed multiplier
SCROLL_DEADZONE = 5             # Pixels of vertical movement to ignore to avoid micro-scroll jitter

# State debouncing (in frames or seconds)
CLICK_DEBOUNCE_FRAMES = 10       # Number of frames to wait before registering another click
