import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import Counter
from typing import Tuple, Optional, Dict, Any, List
import config

class GestureDetector:
    def __init__(self) -> None:
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Debounce and State Machine variables for thumb-based gestures
        self.gesture_history: List[str] = []
        
        # Index Pinch states (Left Click / Right Click / Scroll Mode / Drag & Drop)
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import Counter
from typing import Tuple, Optional, Dict, Any, List
import config

class GestureDetector:
    def __init__(self) -> None:
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Debounce and State Machine variables for thumb-based gestures
        self.gesture_history: List[str] = []
        
        # Index Pinch states (Left Click / Right Click / Scroll Mode / Drag & Drop)
        self.pinch_active: bool = False
        self.pinch_start_time: float = 0.0
        self.pinch_start_pos: Optional[Tuple[float, float]] = None
        self.in_scroll_mode: bool = False
        self.is_dragging: bool = False
        
        # Double Click gesture state
        self.double_click_active: bool = False
        self.right_pinch_active: bool = False
        self.last_release_time: float = 0.0
        
        # Pause Mouse state
        self.mouse_control_disabled: bool = False
        
        # Tap state machine for Left / Right clicks
        self.tap_count: int = 0
        
        # Hand tracking persistence across frames
        self.active_hand_label: Optional[str] = None
        
        # Keep track of previous index tip position and time for velocity calculation
        self.prev_index_pos: Optional[Tuple[float, float]] = None
        self.prev_time: float = 0.0

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Any], Optional[str], Optional[Any]]:
        """
        Processes a BGR image frame, detects hand landmarks.
        Returns:
            landmarks: active hand landmarks object if detected, else None
            hand_label: 'Left' or 'Right' hand label, else None
            results: full MediaPipe process results object
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Find if our active hand is still in the results
            active_idx = None
            if self.active_hand_label is not None:
                for idx, handedness in enumerate(results.multi_handedness):
                    if handedness.classification[0].label == self.active_hand_label:
                        if handedness.classification[0].score > 0.5:
                            active_idx = idx
                            break
            
            # If our active hand was not found or has low confidence, choose the best one
            if active_idx is None:
                best_idx = 0
                best_score = -1.0
                for idx, handedness in enumerate(results.multi_handedness):
                    score = handedness.classification[0].score
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                active_idx = best_idx
                
            self.active_hand_label = results.multi_handedness[active_idx].classification[0].label
            
            landmarks = results.multi_hand_landmarks[active_idx]
            hand_label = self.active_hand_label
            return landmarks, hand_label, results
        
        # Reset active hand tracking if no hand is detected
        self.active_hand_label = None
        return None, None, None

    def draw_landmarks(self, frame: np.ndarray, landmarks: Any, gesture_data: Optional[Dict[str, Any]] = None) -> None:
        """Draws detected hand landmarks and connections on the frame, with custom visual feedback for pinches."""
        if landmarks:
            # Draw standard MediaPipe skeleton first
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Custom premium visual feedback: bigger dots, extra midpoint guide dots, glowing finger tips
            try:
                h, w, c = frame.shape
                lms = landmarks.landmark
                thumb_tip = lms[4]
                index_tip = lms[8]
                middle_tip = lms[12]
                ring_tip = lms[16]
                pinky_tip = lms[20]
                
                # Convert normalized coordinates to pixel coordinates
                thumb_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_px = (int(index_tip.x * w), int(index_tip.y * h))
                midpoint_px = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                
                # Draw intermediate guidance dots along the connection path
                quarter1_px = (int(thumb_px[0] * 0.5 + midpoint_px[0] * 0.5), int(thumb_px[1] * 0.5 + midpoint_px[1] * 0.5))
                quarter3_px = (int(index_px[0] * 0.5 + midpoint_px[0] * 0.5), int(index_px[1] * 0.5 + midpoint_px[1] * 0.5))
                
                is_pinched = False
                if gesture_data:
                    is_pinched = gesture_data.get("pinch_active", False)
                else:
                    wrist = lms[0]
                    middle_mcp = lms[9]
                    hand_scale = self.get_distance(wrist, middle_mcp)
                    if hand_scale > 0:
                        dist_ti = self.get_distance(thumb_tip, index_tip) / hand_scale
                        is_pinched = dist_ti < config.PINCH_THRESHOLD
                
                # Define glowing colors (BGR)
                color_cyan = (255, 255, 0)
                color_magenta = (255, 0, 255)
                color_green_glow = (0, 255, 128)
                color_green_solid = (0, 255, 0)
                color_grey = (200, 200, 200)
                color_orange = (0, 165, 255)
                color_purple = (180, 0, 180)
                color_blue = (255, 100, 0)

                if is_pinched:
                    # Glowing thick line when pinched (in touch)
                    cv2.line(frame, thumb_px, index_px, color_green_solid, 6, cv2.LINE_AA)
                    
                    # Large glowing midpoint dots
                    cv2.circle(frame, midpoint_px, 18, color_green_glow, -1, cv2.LINE_AA)
                    cv2.circle(frame, midpoint_px, 26, color_green_glow, 3, cv2.LINE_AA)
                    
                    # Large glowing finger tip dots for pinch
                    cv2.circle(frame, thumb_px, 16, color_green_solid, -1, cv2.LINE_AA)
                    cv2.circle(frame, thumb_px, 22, color_green_solid, 3, cv2.LINE_AA)
                    cv2.circle(frame, index_px, 16, color_green_solid, -1, cv2.LINE_AA)
                    cv2.circle(frame, index_px, 22, color_green_solid, 3, cv2.LINE_AA)
                    
                    # Intermediate dots also glow
                    cv2.circle(frame, quarter1_px, 11, color_green_glow, -1, cv2.LINE_AA)
                    cv2.circle(frame, quarter3_px, 11, color_green_glow, -1, cv2.LINE_AA)
                else:
                    # Faint connecting guideline when not pinched
                    cv2.line(frame, thumb_px, index_px, color_grey, 2, cv2.LINE_AA)
                    
                    # Midpoint "between dot" - larger, layered circle
                    cv2.circle(frame, midpoint_px, 13, (120, 120, 120), -1, cv2.LINE_AA)
                    cv2.circle(frame, midpoint_px, 18, (160, 160, 160), 3, cv2.LINE_AA)
                    
                    # Thumb & Index tips - styled larger dots
                    cv2.circle(frame, thumb_px, 15, color_cyan, -1, cv2.LINE_AA)
                    cv2.circle(frame, thumb_px, 21, color_cyan, 3, cv2.LINE_AA)
                    cv2.circle(frame, index_px, 15, color_magenta, -1, cv2.LINE_AA)
                    cv2.circle(frame, index_px, 21, color_magenta, 3, cv2.LINE_AA)
                    
                    # Intermediate guidance dots
                    cv2.circle(frame, quarter1_px, 9, (180, 180, 180), -1, cv2.LINE_AA)
                    cv2.circle(frame, quarter3_px, 9, (180, 180, 180), -1, cv2.LINE_AA)
                
                # Draw other finger tips larger and more styled (Middle, Ring, Pinky)
                other_tips = [
                    ((int(middle_tip.x * w), int(middle_tip.y * h)), color_orange),
                    ((int(ring_tip.x * w), int(ring_tip.y * h)), color_purple),
                    ((int(pinky_tip.x * w), int(pinky_tip.y * h)), color_blue)
                ]
                for tip_px, color in other_tips:
                    cv2.circle(frame, tip_px, 14, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, tip_px, 19, color, 3, cv2.LINE_AA)
                    
            except Exception as e:
                print(f"[Detector] Draw visual feedback error: {e}")

    def get_distance(self, lm1: Any, lm2: Any) -> float:
        """Calculates Euclidean distance between two landmarks in 2D (x, y)."""
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

    def is_extended(self, lms: Any, tip_idx: int, mcp_idx: int, hand_scale: float, threshold: float) -> bool:
        """Determines if a finger is extended based on knuckle-to-tip distance scaled by hand size."""
        return (self.get_distance(lms[mcp_idx], lms[tip_idx]) / hand_scale) > threshold

    def get_gesture(self, landmarks: Optional[Any], hand_label: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Identifies the hand gesture (Move Cursor, Left Click, Right Click, Scroll Mode, Pause)
        using a state machine based on thumb-pinches, finger extensions, adaptive scaling, 
        and majority-vote sliding window debouncing.
        
        RESTRICTION: Controls are enabled only for the user's physical right hand.
        """
        # 0. Handle hand lost or none detected
        if not landmarks or not hand_label:
            # Reset transient gesture state when hand is lost
            self.gesture_history.clear()
            self.pinch_active = False
            self.in_scroll_mode = False
            self.is_dragging = False
            self.double_click_active = False
            self.right_pinch_active = False
            self.last_release_time = 0.0
            self.mouse_control_disabled = False
            self.tap_count = 0
            self.prev_index_pos = None
            self.prev_time = 0.0
            return "No Hand", {"confidence": 0.0}

        # 0.1 Check for Right Hand Only constraint.
        # Since the camera feed is mirrored horizontally, a physical right hand is classified as "Left" by MediaPipe.
        is_physical_right = False
        if hand_label == "Left" and config.MIRROR_HAND_LABEL:
            is_physical_right = True
        elif hand_label == "Right" and not config.MIRROR_HAND_LABEL:
            is_physical_right = True
            
        if config.RIGHT_HAND_ONLY and not is_physical_right:
            # Clear states and ignore left hand inputs completely
            self.gesture_history.clear()
            self.pinch_active = False
            self.in_scroll_mode = False
            self.is_dragging = False
            self.double_click_active = False
            self.right_pinch_active = False
            self.last_release_time = 0.0
            self.mouse_control_disabled = False
            self.tap_count = 0
            self.prev_index_pos = None
            self.prev_time = 0.0
            return "No Hand", {"confidence": 0.0}

        lms = landmarks.landmark
        current_time = time.time()
        
        # 1. Calculate hand scale (distance between wrist and middle finger MCP)
        wrist = lms[0]
        middle_mcp = lms[9]
        hand_scale = self.get_distance(wrist, middle_mcp)
        if hand_scale == 0:
            hand_scale = 0.1  # Prevent division by zero
        
        # 2. Get finger tip landmarks
        thumb_tip = lms[4]
        index_tip = lms[8]
        middle_tip = lms[12]
        ring_tip = lms[16]
        pinky_tip = lms[20]
        
        # 3. Calculate normalized distance for thumb-index pinch
        dist_thumb_index = self.get_distance(thumb_tip, index_tip) / hand_scale
        dist_thumb_ring = self.get_distance(thumb_tip, ring_tip) / hand_scale
        
        # 4. Calculate finger extension states using knuckle-to-tip ratio
        thumb_extended = self.is_extended(lms, 4, 2, hand_scale, 0.35)
        index_extended = self.is_extended(lms, 8, 5, hand_scale, 0.5)
        middle_extended = self.is_extended(lms, 12, 9, hand_scale, 0.5)
        ring_extended = self.is_extended(lms, 16, 13, hand_scale, 0.5)
        pinky_extended = self.is_extended(lms, 20, 17, hand_scale, 0.4)
        
        # 5. Calculate velocity of index tip
        velocity = 0.0
        if self.prev_index_pos is not None and self.prev_time > 0:
            dt = current_time - self.prev_time
            if dt > 0:
                dx = index_tip.x - self.prev_index_pos[0]
                dy = index_tip.y - self.prev_index_pos[1]
                velocity = math.sqrt(dx*dx + dy*dy) / dt
        
        self.prev_index_pos = (index_tip.x, index_tip.y)
        self.prev_time = current_time

        # 6. Raw Gesture Classification
        raw_gesture = "Move Cursor"
        raw_confidence = 1.0
        
        # Pause/Resume gestures:
        # Open hand: all fingers extended
        is_open_hand = (thumb_extended and index_extended and middle_extended and 
                        ring_extended and pinky_extended)
        
        # Closed fist: all fingers folded
        is_fist = (not thumb_extended and not index_extended and not middle_extended and 
                   not ring_extended and not pinky_extended)
        
        if config.PAUSE_GESTURE_ENABLED and is_fist:
            self.mouse_control_disabled = True
        elif is_open_hand:
            self.mouse_control_disabled = False

        # Pinch threshold using hysteresis (Touch / Release thresholds)
        index_pinch_threshold = config.PINCH_RELEASE_THRESHOLD if self.pinch_active else config.PINCH_THRESHOLD
        is_pinched = dist_thumb_index < index_pinch_threshold

        # 7. Debounce using sliding window majority vote
        if self.mouse_control_disabled:
            if is_open_hand:
                self.mouse_control_disabled = False
                debounced_gesture = "Move Cursor"
            else:
                debounced_gesture = "Pause Mouse"
        else:
            if is_pinched:
                raw_gesture = "Index Pinch"
                raw_confidence = max(0.0, min(1.0, 1.0 - (dist_thumb_index / index_pinch_threshold)))
            else:
                raw_gesture = "Move Cursor"
                raw_confidence = 1.0

            self.gesture_history.append(raw_gesture)
            if len(self.gesture_history) > config.GESTURE_DEBOUNCE_FRAMES:
                self.gesture_history.pop(0)
                
            counter = Counter(self.gesture_history)
            debounced_gesture = counter.most_common(1)[0][0]
        
        gesture_event = "Move Cursor"
        is_pinched_debounced = (debounced_gesture == "Index Pinch")
        
        # 8. State Machine for gesture mapping (Clicks on release, Scroll/Drag on hold)
        if debounced_gesture == "Pause Mouse":
            self.pinch_active = False
            self.in_scroll_mode = False
            self.is_dragging = False
            self.tap_count = 0
            self.double_click_active = False
            self.right_pinch_active = False
            gesture_event = "Pause Mouse"
        else:
            # Check for double click touch (Thumb + Middle)
            dist_thumb_middle = self.get_distance(thumb_tip, middle_tip) / hand_scale
            double_click_threshold = config.DOUBLE_CLICK_RELEASE_THRESHOLD if self.double_click_active else config.DOUBLE_CLICK_THRESHOLD
            is_double_click_pinch = dist_thumb_middle < double_click_threshold
            
            # Check for right click touch (Thumb + Ring)
            dist_thumb_ring = self.get_distance(thumb_tip, ring_tip) / hand_scale
            right_click_threshold = config.RIGHT_CLICK_RELEASE_THRESHOLD if self.right_pinch_active else config.RIGHT_CLICK_THRESHOLD
            is_right_click_pinch = dist_thumb_ring < right_click_threshold

            if is_double_click_pinch:
                if not self.double_click_active:
                    self.double_click_active = True
                    gesture_event = "Double Click"
                    # Reset pinch/scroll/drag states on double click touch
                    self.pinch_active = False
                    self.is_dragging = False
                    self.in_scroll_mode = False
                    self.right_pinch_active = False
                else:
                    gesture_event = "Move Cursor"
            else:
                if dist_thumb_middle > config.RELEASE_THRESHOLD:
                    self.double_click_active = False

                if is_right_click_pinch:
                    if not self.right_pinch_active:
                        self.right_pinch_active = True
                        gesture_event = "Right Click"
                        # Reset pinch/scroll/drag states
                        self.pinch_active = False
                        self.is_dragging = False
                        self.in_scroll_mode = False
                    else:
                        gesture_event = "Move Cursor"
                else:
                    if dist_thumb_ring > config.RELEASE_THRESHOLD:
                        self.right_pinch_active = False

                    # Process thumb-index pinch if middle/ring pinches are not active
                    if not is_double_click_pinch and not is_right_click_pinch:
                        if is_pinched_debounced:
                            if not self.pinch_active:
                                # Pinch just started (index and thumb got in touch with between dot)
                                self.pinch_active = True
                                self.pinch_start_time = current_time
                                self.pinch_start_pos = (index_tip.x, index_tip.y)
                                self.in_scroll_mode = False
                                self.is_dragging = False
                                
                                # Check if this is a rapid double tap (for Right Click)
                                time_since_last_release = current_time - self.last_release_time
                                if time_since_last_release < config.DOUBLE_TAP_MAX_WINDOW:
                                    gesture_event = "Right Click"
                                else:
                                    gesture_event = "Left Click"
                            else:
                                # Pinch is held
                                if self.is_dragging:
                                    gesture_event = "Drag"
                                elif self.in_scroll_mode:
                                    gesture_event = "Scroll Mode"
                                else:
                                    elapsed = current_time - self.pinch_start_time
                                    dy = index_tip.y - self.pinch_start_pos[1]
                                    dx = index_tip.x - self.pinch_start_pos[0]
                                    dist_moved = math.sqrt(dx*dx + dy*dy)
                                    
                                    if elapsed > config.DRAG_START_DELAY:
                                        self.is_dragging = True
                                        gesture_event = "Drag Start"
                                    elif dist_moved > config.SCROLL_START_THRESHOLD:
                                        self.in_scroll_mode = True
                                        gesture_event = "Scroll Mode"
                                    else:
                                        gesture_event = "Move Cursor"
                        else:
                            # Pinch is not active
                            if self.pinch_active:
                                # Pinch was just released
                                self.pinch_active = False
                                self.last_release_time = current_time
                                if self.is_dragging:
                                    self.is_dragging = False
                                    gesture_event = "Drag End"
                                elif self.in_scroll_mode:
                                    self.in_scroll_mode = False
                                    gesture_event = "Move Cursor"
                                else:
                                    gesture_event = "Move Cursor"
                            else:
                                gesture_event = "Move Cursor"

            # Screenshot Override: Index, Middle, Ring extended, Thumb and Pinky folded
            is_screenshot_gesture = (index_extended and middle_extended and ring_extended and 
                                     not thumb_extended and not pinky_extended)
            if is_screenshot_gesture:
                gesture_event = "Screenshot"

        gesture_data = {
            "index_tip": (index_tip.x, index_tip.y),
            "thumb_tip": (thumb_tip.x, thumb_tip.y),
            "ring_tip": (ring_tip.x, ring_tip.y),
            "middle_tip": (middle_tip.x, middle_tip.y),
            "dist_thumb_index": dist_thumb_index,
            "dist_thumb_ring": dist_thumb_ring,
            "hand_scale": hand_scale,
            "velocity": velocity,
            "confidence": raw_confidence,
            "is_dragging": self.is_dragging,
            "pinch_active": self.pinch_active,
            "finger_states": {
                "Thumb": thumb_extended,
                "Index": index_extended,
                "Middle": middle_extended,
                "Ring": ring_extended,
                "Pinky": pinky_extended
            }
        }
        
        return gesture_event, gesture_data
