import cv2
import mediapipe as mp
import numpy as np
import math
import config

class GestureDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """
        Processes a BGR image frame, detects hand landmarks.
        Returns:
            landmarks: list of landmarks if detected, else None
            hand_label: 'Left' or 'Right' hand if detected, else None
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # We only track the first hand detected
            landmarks = results.multi_hand_landmarks[0]
            # Get hand label (Left or Right)
            hand_label = results.multi_handedness[0].classification[0].label
            return landmarks, hand_label, results
        
        return None, None, None

    def draw_landmarks(self, frame, landmarks):
        """Draws detected hand landmarks on the image frame."""
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

    def get_distance(self, lm1, lm2):
        """Calculates Euclidean distance between two landmarks in 3D (x, y, z)."""
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

    def get_gesture(self, landmarks, hand_label):
        """
        Identifies the hand gesture based on landmark positions.
        Returns a string representing the detected gesture and supplementary data.
        """
        if not landmarks:
            return "No Hand", {}

        lms = landmarks.landmark
        
        # 1. Calculate hand scale (distance between wrist and middle finger MCP)
        # This makes thresholds scale-invariant (works whether hand is close or far)
        wrist = lms[0]
        middle_mcp = lms[9]
        hand_scale = self.get_distance(wrist, middle_mcp)
        if hand_scale == 0:
            hand_scale = 0.1 # Prevent division by zero
        
        # 2. Get finger tip and joint landmarks
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        
        index_tip = lms[8]
        index_pip = lms[6]
        index_mcp = lms[5]
        
        middle_tip = lms[12]
        middle_pip = lms[10]
        
        ring_tip = lms[16]
        ring_pip = lms[14]
        
        pinky_tip = lms[20]
        pinky_pip = lms[18]
        
        # 3. Determine if fingers are extended (Up) or folded (Down)
        # Using Y-coordinates (Y decreases upwards in screen space)
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y
        
        # Thumb up calculation: thumb tip is further out from index MCP (horizontal check)
        # Depending on Left vs Right hand
        if hand_label == "Right":
            thumb_up = thumb_tip.x < thumb_ip.x
        else:
            thumb_up = thumb_tip.x > thumb_ip.x

        # 4. Compute normalized distances
        thumb_index_dist = self.get_distance(thumb_tip, index_tip) / hand_scale
        thumb_middle_dist = self.get_distance(thumb_tip, middle_tip) / hand_scale
        index_middle_dist = self.get_distance(index_tip, middle_tip) / hand_scale
        
        # Save info for rendering or debugging
        gesture_data = {
            "index_tip": (index_tip.x, index_tip.y),
            "middle_tip": (middle_tip.x, middle_tip.y),
            "thumb_index_dist": thumb_index_dist,
            "thumb_middle_dist": thumb_middle_dist,
            "index_middle_dist": index_middle_dist,
            "hand_scale": hand_scale
        }

        # 5. Gesture Classification Logic
        # Closed Fist (Drag & Drop): Index, Middle, Ring, Pinky all folded down
        if not (index_up or middle_up or ring_up or pinky_up):
            return "Fist (Drag)", gesture_data

        # Two Fingers Up (Scroll): Index and Middle up, Ring and Pinky down
        if index_up and middle_up and not ring_up and not pinky_up:
            # Check if they are squeezed together (Double Click)
            if index_middle_dist < config.DOUBLE_CLICK_THRESHOLD:
                return "Double Click", gesture_data
            else:
                return "Two Fingers Up (Scroll)", gesture_data

        # Index and Thumb pinch (Left Click) while index is up and middle/ring/pinky are folded
        if index_up and not middle_up and not ring_up and not pinky_up:
            if thumb_index_dist < config.PINCH_THRESHOLD:
                return "Left Click", gesture_data
            else:
                return "Move Cursor", gesture_data

        # Thumb and Middle pinch (Right Click)
        # Can happen when index and middle are up, or just middle is up
        if middle_up and not ring_up and not pinky_up:
            if thumb_middle_dist < config.RIGHT_CLICK_THRESHOLD:
                return "Right Click", gesture_data

        # Default fallback: if index is up, move cursor anyway
        if index_up:
            return "Move Cursor", gesture_data

        return "Unknown Gesture", gesture_data
