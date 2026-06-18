import pyautogui
import time
import config

# Disable PyAutoGUI delay for faster responsiveness
pyautogui.PAUSE = 0.001

class MouseController:
    def __init__(self):
        # Current smoothed screen coordinates
        self.smooth_x = None
        self.smooth_y = None
        
        # Gestures states and debouncing
        self.click_cooldown_until = 0.0
        self.is_dragging = False
        
        # Scroll tracking
        self.is_scrolling = False
        self.last_scroll_y = None
        
        # Offset to prevent cursor jump when transitioning from Index Tip to Fist Drag
        self.fist_offset_x = 0.0
        self.fist_offset_y = 0.0
        self.was_fist = False

    def transform_coordinates(self, x, y):
        """
        Maps normalized camera coordinates (x, y) to screen pixels (screen_x, screen_y)
        using the config's tracking sub-region and sensitivity.
        """
        # Calculate tracking box center and sizes
        x_center = (config.TRACKING_BOX_LEFT + config.TRACKING_BOX_RIGHT) / 2
        y_center = (config.TRACKING_BOX_TOP + config.TRACKING_BOX_BOTTOM) / 2
        
        box_width = config.TRACKING_BOX_RIGHT - config.TRACKING_BOX_LEFT
        box_height = config.TRACKING_BOX_BOTTOM - config.TRACKING_BOX_TOP
        
        # Shrink tracking box boundaries based on sensitivity
        half_width = box_width / (2 * config.SENSITIVITY)
        half_height = box_height / (2 * config.SENSITIVITY)
        
        left = x_center - half_width
        right = x_center + half_width
        top = y_center - half_height
        bottom = y_center + half_height
        
        # Map to [0.0, 1.0] range
        mapped_x = (x - left) / (right - left)
        mapped_y = (y - top) / (bottom - top)
        
        # Clamp coordinates to screen boundaries
        mapped_x = max(0.0, min(1.0, mapped_x))
        mapped_y = max(0.0, min(1.0, mapped_y))
        
        # Convert to pixels
        screen_x = int(mapped_x * config.SCREEN_WIDTH)
        screen_y = int(mapped_y * config.SCREEN_HEIGHT)
        
        return screen_x, screen_y

    def smooth_coordinates(self, target_x, target_y):
        """Applies Exponential Moving Average (EMA) to smooth out cursor movement."""
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = target_x
            self.smooth_y = target_y
        else:
            alpha = config.SMOOTHING_ALPHA
            self.smooth_x = alpha * target_x + (1 - alpha) * self.smooth_x
            self.smooth_y = alpha * target_y + (1 - alpha) * self.smooth_y
            
        return int(self.smooth_x), int(self.smooth_y)

    def execute_action(self, gesture, landmarks, hand_label):
        """
        Translates gestures into system mouse actions.
        Returns:
            cursor_pos: tuple (x, y) representing current cursor position or None
        """
        current_time = time.time()
        
        if gesture == "No Hand":
            # Release drag if hand leaves camera view
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
            self.is_scrolling = False
            self.was_fist = False
            return None

        lms = landmarks.landmark
        
        # 1. Handle Scrolling State
        if gesture == "Two Fingers Up (Scroll)":
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                
            self.was_fist = False
            
            # Use average of index and middle finger tips
            current_scroll_y = (lms[8].y + lms[12].y) / 2
            
            if not self.is_scrolling:
                self.is_scrolling = True
                self.last_scroll_y = current_scroll_y
            else:
                dy = current_scroll_y - self.last_scroll_y
                # If vertical movement exceeds a tiny threshold, perform scroll
                if abs(dy) > (config.SCROLL_DEADZONE / config.CAM_HEIGHT):
                    # Negative dy means moving UP (towards screen top), scroll UP (positive pyautogui scroll)
                    scroll_amount = -dy * config.SCROLL_SENSITIVITY * config.CAM_HEIGHT
                    pyautogui.scroll(int(scroll_amount))
                    self.last_scroll_y = current_scroll_y
            return None

        # Reset scrolling state if not in scroll gesture
        self.is_scrolling = False

        # 2. Determine tracking coordinates
        # Default is index tip (8)
        tracking_lm = lms[8]
        
        if gesture == "Fist (Drag)":
            # For Fist, track landmark 9 (middle MCP) to keep it stable
            tracking_lm = lms[9]
            
            if not self.was_fist:
                # First frame transitioning into Fist: compute offset from previous smooth position
                # so the cursor doesn't jump
                if self.smooth_x is not None and self.smooth_y is not None:
                    # Target screen coordinates of landmark 9
                    target_x, target_y = self.transform_coordinates(tracking_lm.x, tracking_lm.y)
                    self.fist_offset_x = self.smooth_x - target_x
                    self.fist_offset_y = self.smooth_y - target_y
                self.was_fist = True
        else:
            self.was_fist = False

        # Transform and smooth coordinates
        target_x, target_y = self.transform_coordinates(tracking_lm.x, tracking_lm.y)
        
        if gesture == "Fist (Drag)":
            # Apply offset to prevent jump
            target_x += self.fist_offset_x
            target_y += self.fist_offset_y
            
        smooth_x, smooth_y = self.smooth_coordinates(target_x, target_y)

        # 3. Handle Drag & Drop state
        if gesture == "Fist (Drag)":
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
            pyautogui.moveTo(smooth_x, smooth_y)
            return smooth_x, smooth_y
        
        # Release drag if not in fist gesture anymore
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False

        # 4. Handle Movement and Clicks
        if gesture in ["Move Cursor", "Left Click", "Right Click", "Double Click"]:
            # Move cursor
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # Click action execution with debounce
            if current_time > self.click_cooldown_until:
                if gesture == "Left Click":
                    pyautogui.click()
                    self.click_cooldown_until = current_time + (config.CLICK_DEBOUNCE_FRAMES / config.FRAME_RATE)
                elif gesture == "Right Click":
                    pyautogui.rightClick()
                    self.click_cooldown_until = current_time + (config.CLICK_DEBOUNCE_FRAMES / config.FRAME_RATE)
                elif gesture == "Double Click":
                    pyautogui.doubleClick()
                    self.click_cooldown_until = current_time + (config.CLICK_DEBOUNCE_FRAMES / config.FRAME_RATE)
            
            return smooth_x, smooth_y

        return smooth_x, smooth_y
