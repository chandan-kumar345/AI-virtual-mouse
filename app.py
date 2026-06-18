import cv2
import time
import sys
import pyautogui

import config
from gesture_detector import GestureDetector
from mouse_controller import MouseController

# Helper function to draw text with a drop-shadow for premium readability
def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255, 255, 255), thickness=1):
    x, y = pos
    # Shadow (black)
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Foreground text
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def main():
    print("Initializing AI Virtual Mouse...")
    print(f"Screen Resolution: {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
    print(f"Webcam Target Resolution: {config.CAM_WIDTH}x{config.CAM_HEIGHT}")
    
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {config.CAMERA_INDEX}.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    
    # Instantiate detector and controller
    detector = GestureDetector()
    controller = MouseController()
    
    # Frame rate calculation variables
    prev_time = time.time()
    fps = 0
    
    # Active tracking box pixels coordinates
    box_left = int(config.TRACKING_BOX_LEFT * config.CAM_WIDTH)
    box_right = int(config.TRACKING_BOX_RIGHT * config.CAM_WIDTH)
    box_top = int(config.TRACKING_BOX_TOP * config.CAM_HEIGHT)
    box_bottom = int(config.TRACKING_BOX_BOTTOM * config.CAM_HEIGHT)
    
    # Window name
    window_name = "AI Virtual Mouse - Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("\nVirtual Mouse is running!")
    print("Move your hand in front of the camera inside the green tracking box.")
    print("Press 'Q' inside the camera window to Exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break
                
            # Mirror the frame horizontally for natural user mapping
            frame = cv2.flip(frame, 1)
            
            # Process frame for landmarks
            landmarks, hand_label, results = detector.process_frame(frame)
            
            # Track and execute gestures
            gesture = "No Hand"
            gesture_data = {}
            cursor_pos = None
            
            if landmarks:
                gesture, gesture_data = detector.get_gesture(landmarks, hand_label)
                
                # Execute mouse actions and move the cursor
                try:
                    cursor_pos = controller.execute_action(gesture, landmarks, hand_label)
                except pyautogui.FailSafeException:
                    # Gracefully handle PyAutoGUI failsafe (mouse moved to corner)
                    print("PyAutoGUI Failsafe triggered: Cursor moved to a corner. Recovery mode active.")
                    # Sleep briefly to give the user time to move hand away from corner
                    time.sleep(0.5)
                
                # Draw landmarks on the frame
                detector.draw_landmarks(frame, landmarks)
            else:
                # No hands in screen, make sure to reset state
                controller.execute_action("No Hand", None, None)
            
            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # RENDER OVERLAYS
            
            # 1. Semi-transparent header panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (config.CAM_WIDTH, 60), (25, 25, 25), -1)
            # 2. Semi-transparent footer panel for shortcut legends
            cv2.rectangle(overlay, (0, config.CAM_HEIGHT - 35), (config.CAM_WIDTH, config.CAM_HEIGHT), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw tracking box boundary (Active zone)
            # Color is light green when active tracking is working
            box_color = (0, 255, 128) if landmarks else (100, 100, 100)
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)
            draw_text(frame, "Active Tracking Zone", (box_left + 5, box_top - 8), scale=0.45, color=box_color, thickness=1)
            
            # Choose gesture label text color for quick visual status check
            gesture_colors = {
                "No Hand": (150, 150, 150),
                "Move Cursor": (255, 200, 0),
                "Left Click": (0, 255, 0),
                "Right Click": (0, 0, 255),
                "Double Click": (0, 255, 255),
                "Fist (Drag)": (0, 120, 255),
                "Two Fingers Up (Scroll)": (255, 0, 255),
                "Unknown Gesture": (0, 0, 255)
            }
            g_color = gesture_colors.get(gesture, (255, 255, 255))
            
            # HUD text output
            draw_text(frame, f"Gesture: {gesture}", (15, 25), scale=0.6, color=g_color, thickness=2)
            draw_text(frame, f"Hand: {hand_label if hand_label else 'N/A'}", (280, 25), scale=0.55, color=(200, 255, 200), thickness=1)
            draw_text(frame, f"FPS: {fps:.1f}", (config.CAM_WIDTH - 110, 25), scale=0.55, color=(0, 255, 255), thickness=1)
            
            # Show cursor position if active
            if cursor_pos:
                cx, cy = cursor_pos
                draw_text(frame, f"Cursor: ({cx}, {cy})", (15, 48), scale=0.45, color=(200, 200, 200), thickness=1)
            
            # Legends / Shortcuts footer
            legends = "Fist: Drag | 2-Fingers: Scroll | Index+Thumb: L-Click | Idx+Mid: D-Click | Q: Exit"
            draw_text(frame, legends, (15, config.CAM_HEIGHT - 12), scale=0.42, color=(180, 180, 180), thickness=1)
            
            # Show image frame
            cv2.imshow(window_name, frame)
            
            # Exit program when 'Q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exit signal received. Terminating...")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Terminating...")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. All windows closed. Goodbye!")

if __name__ == "__main__":
    main()
