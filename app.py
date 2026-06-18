import cv2
import time
import sys
import numpy as np
import threading
from typing import Tuple, Optional, Any
import pyautogui

import config
from gesture_detector import GestureDetector
from mouse_controller import MouseController

class ThreadedCamera:
    """
    Asynchronously reads frames from VideoCapture on a background thread
    to decouple frame capture latency from main-thread calculations.
    Automatically handles camera reconnects in the background.
    """
    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src: int = src
        self.width: int = width
        self.height: int = height
        
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.grabbed: bool = False
        self.frame: Optional[np.ndarray] = None
        self.started: bool = False
        self.is_connected: bool = self.cap.isOpened()
        
        self.read_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> 'ThreadedCamera':
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self) -> None:
        consecutive_failures = 0
        while self.started:
            if not self.is_connected:
                time.sleep(1.0)
                with self.read_lock:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    if self.cap.isOpened():
                        self.is_connected = True
                        consecutive_failures = 0
                continue
                
            grabbed, frame = self.cap.read()
            if not grabbed:
                consecutive_failures += 1
                if consecutive_failures > 30:  # ~1s of failures
                    self.is_connected = False
            else:
                consecutive_failures = 0
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self) -> None:
        self.started = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], 
              font: int = cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.5, 
              color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    """Helper function to draw text with a high-contrast black drop-shadow."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_corners(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                 color: Tuple[int, int, int], thickness: int, r: int) -> None:
    """Draws premium bracket-style corners for the active tracking box."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + r, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + r), color, thickness)
    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + r), color, thickness)
    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + r, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - r), color, thickness)
    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - r), color, thickness)


def main() -> None:
    # Clear pinch debug log
    try:
        open("pinch_debug.txt", "w").close()
    except Exception:
        pass
        
    print("Initializing AI Virtual Mouse...")
    print(f"Screen Resolution: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
    print(f"Webcam Target Resolution: {config.CAM_WIDTH}x{config.CAM_HEIGHT}")
    
    # Initialize threaded camera capture
    cam = ThreadedCamera(src=config.CAMERA_INDEX, width=config.CAM_WIDTH, height=config.CAM_HEIGHT)
    cam.start()
    
    detector = GestureDetector()
    controller = MouseController()
    
    prev_time = time.time()
    fps = 0.0
    
    box_left = int(config.TRACKING_BOX_LEFT * config.CAM_WIDTH)
    box_right = int(config.TRACKING_BOX_RIGHT * config.CAM_WIDTH)
    box_top = int(config.TRACKING_BOX_TOP * config.CAM_HEIGHT)
    box_bottom = int(config.TRACKING_BOX_BOTTOM * config.CAM_HEIGHT)
    
    window_name = "AI Virtual Mouse - Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("\nVirtual Mouse is running!")
    print("Move your hand in front of the camera inside the green tracking box.")
    print("Press 'Q' inside the camera window to Exit.")

    try:
        while True:
            # Check camera connection state
            if not cam.is_connected:
                # Reconnection Screen Overlay
                fail_frame = np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), dtype=np.uint8)
                cv2.rectangle(fail_frame, (50, 180), (config.CAM_WIDTH - 50, 300), (30, 30, 30), -1)
                cv2.rectangle(fail_frame, (50, 180), (config.CAM_WIDTH - 50, 300), (0, 0, 255), 2)
                draw_text(fail_frame, "CAMERA DISCONNECTED", (config.CAM_WIDTH // 2 - 140, 230), scale=0.7, color=(0, 0, 255), thickness=2)
                draw_text(fail_frame, "Attempting to reconnect...", (config.CAM_WIDTH // 2 - 120, 265), scale=0.5, color=(200, 200, 200), thickness=1)
                cv2.imshow(window_name, fail_frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                continue

            ret, frame = cam.read()
            if not ret or frame is None:
                # Frame lag fallback
                time.sleep(0.01)
                continue
                
            # Mirror horizontally for natural tracking
            frame = cv2.flip(frame, 1)
            
            # Process landmarks and gestures
            landmarks, hand_label, results = detector.process_frame(frame)
            
            gesture = "No Hand"
            confidence = 0.0
            gesture_confidence = 0.0
            cursor_pos = None
            dist_ti = None
            
            if landmarks:
                # Expose confidence rating
                try:
                    confidence = results.multi_handedness[0].classification[0].score
                except Exception:
                    confidence = 1.0

                gesture, gesture_data = detector.get_gesture(landmarks, hand_label)
                gesture_confidence = gesture_data.get("confidence", 0.0)
                dist_ti = gesture_data.get("dist_thumb_index", None)
                
                # Execute mouse movement/clicks
                try:
                    cursor_pos = controller.execute_action(gesture, landmarks, hand_label)
                except pyautogui.FailSafeException:
                    print("[Failsafe] Mouse cursor moved to corner. Recovery activated.")
                    controller.reset_tracking()
                    time.sleep(0.5)
                
                detector.draw_landmarks(frame, landmarks)
            else:
                controller.execute_action("No Hand", None, None)
            
            # FPS Calculation
            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time
            
            # PREMIUM HUD RENDERING
            
            # 1. Top Panel (Header Bar) - Glassmorphism style
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (config.CAM_WIDTH, 60), (35, 30, 30), -1)
            # 2. Bottom Panel (Footer Legends)
            cv2.rectangle(overlay, (0, config.CAM_HEIGHT - 35), (config.CAM_WIDTH, config.CAM_HEIGHT), (30, 25, 25), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            
            # 3. Blinking Pause Status Badge
            is_paused = controller.is_paused
            state_color = (0, 165, 255) if is_paused else (0, 255, 128)  # Orange vs Bright Green
            state_text = "SYSTEM PAUSED" if is_paused else "SYSTEM ACTIVE"
            
            # Blinking light dot
            if not is_paused or int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (25, 28), 6, state_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (25, 28), 6, (20, 20, 20), 1, cv2.LINE_AA)
            draw_text(frame, state_text, (40, 33), scale=0.55, color=state_color, thickness=2)
            
            # 4. Tracking Quality indicator
            if not landmarks:
                q_text, q_color = "STANDBY", (150, 150, 150)
            elif fps > 25 and confidence > 0.85:
                q_text, q_color = "EXCELLENT", (0, 255, 0)
            elif fps > 18 and confidence > 0.7:
                q_text, q_color = "GOOD", (0, 255, 255)
            else:
                q_text, q_color = "LOW QUALITY", (0, 0, 255)
            
            draw_text(frame, f"Quality: {q_text}", (200, 33), scale=0.5, color=q_color, thickness=2)
            
            # 5. Show Hand confidence bar
            if landmarks:
                # Progress bar for confidence
                bar_x1, bar_y1 = 360, 20
                bar_width = 100
                bar_height = 10
                filled_width = int(gesture_confidence * bar_width)
                # BG box
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_width, bar_y1 + bar_height), (50, 50, 50), -1)
                # Filled box
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled_width, bar_y1 + bar_height), (0, 255, 0), -1)
                # Text
                draw_text(frame, f"Hand: {int(confidence*100)}% Gest: {int(gesture_confidence*100)}%", (bar_x1, bar_y1 - 6), scale=0.4, color=(200, 255, 200), thickness=1)
                
                # Visual pinch distance debugging
                if dist_ti is not None:
                    draw_text(frame, f"Pinch Dist: {dist_ti:.3f} / {config.PINCH_THRESHOLD:.3f}", (360, 43), scale=0.4, color=(200, 255, 200), thickness=1)
            else:
                draw_text(frame, "Gest: 0%", (360, 14), scale=0.4, color=(150, 150, 150), thickness=1)
                
            # 6. FPS Display
            draw_text(frame, f"FPS: {fps:.1f}", (config.CAM_WIDTH - 105, 33), scale=0.55, color=(0, 255, 255), thickness=2)
            
            # 7. Tracking Zone Corner Brackets (Active Box)
            box_color = (0, 255, 128) if landmarks else (120, 120, 120)
            draw_corners(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2, 20)
            draw_text(frame, "Active Tracking Box", (box_left + 5, box_top - 8), scale=0.42, color=box_color, thickness=1)
            
            # 8. Render Gesture Label Card (Center-Bottom Overlay)
            gesture_colors = {
                "No Hand": (120, 120, 120),
                "Move Cursor": (255, 200, 50),
                "Left Click": (0, 255, 0),
                "Right Click": (0, 100, 255),
                "Double Click": (0, 255, 255),
                "Drag Start": (255, 0, 255),
                "Drag": (255, 0, 255),
                "Drop": (0, 255, 0),
                "Scroll Mode": (255, 255, 0),
                "Pause Cursor": (0, 0, 255)
            }
            g_color = gesture_colors.get(gesture, (255, 255, 255))
            
            # Visual click feedback flash
            click_flash_active = (time.time() - controller.last_click_time < 0.25)
            if click_flash_active and controller.last_click_event:
                event_name = controller.last_click_event.replace("_", " ").upper()
                flash_text = f"EVENT: {event_name}!"
                cv2.rectangle(frame, (config.CAM_WIDTH // 2 - 140, 70), (config.CAM_WIDTH // 2 + 140, 105), (0, 255, 0), -1)
                draw_text(frame, flash_text, (config.CAM_WIDTH // 2 - 120, 93), scale=0.6, color=(0, 0, 0), thickness=2)
                
            # Current active gesture text
            draw_text(frame, f"Gesture: {gesture}", (15, config.CAM_HEIGHT - 48), scale=0.55, color=g_color, thickness=2)
            if cursor_pos:
                cx, cy = cursor_pos
                draw_text(frame, f"Screen Pos: ({cx}, {cy})", (240, config.CAM_HEIGHT - 48), scale=0.48, color=(200, 200, 200), thickness=1)
            if hand_label:
                draw_text(frame, f"Hand: {hand_label}", (config.CAM_WIDTH - 120, config.CAM_HEIGHT - 48), scale=0.48, color=(200, 255, 200), thickness=1)
 
            # Legends Footer Description
            legends = "Index: Move | Pinch: Left Click | Double Pinch: Right Click | Hold Pinch: Scroll | Palm: Pause | Q: Exit"
            draw_text(frame, legends, (15, config.CAM_HEIGHT - 12), scale=0.42, color=(180, 180, 180), thickness=1)
            
            # Show image frame
            cv2.imshow(window_name, frame)
            
            # Exit key processing
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[App] Termination requested.")
                break
                
    except KeyboardInterrupt:
        print("\n[App] Keyboard Interrupt detected.")
    finally:
        # Cleanup properly
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera released. All windows closed. Application shutdown cleanly.")

if __name__ == "__main__":
    main()
