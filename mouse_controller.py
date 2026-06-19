import pyautogui
import time
import math
import numpy as np
from typing import Tuple, Optional, Any
import config

# Disable PyAutoGUI delay for faster responsiveness
pyautogui.PAUSE = 0

class OneEuroFilter:
    """
    One Euro Filter for speed-adaptive smoothing.
    """
    def __init__(self, t0: float, x0: float, dx0: float = 0.0, 
                 min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.t_prev: float = t0
        self.x_prev: float = x0
        self.dx_prev: float = dx0
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff

    def __call__(self, t: float, x: float) -> float:
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Filter velocity
        alpha_d = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * self.d_cutoff * dt))
        dx = (x - self.x_prev) / dt
        dx_filtered = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # Calculate adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

        # Filter position
        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev

        # Save states
        self.t_prev = t
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered

        return x_filtered


class KalmanFilter2D:
    """
    Kalman Filter for 2D position and velocity tracking.
    """
    def __init__(self, x0: float, y0: float, q_accel: float = 0.1, r_meas: float = 0.5):
        # State: [x, y, vx, vy]^T
        self.x: np.ndarray = np.array([x0, y0, 0.0, 0.0], dtype=np.float32)
        # Covariance P
        self.P: np.ndarray = np.eye(4, dtype=np.float32) * 10.0
        # Measurement matrix H (we only measure position)
        self.H: np.ndarray = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        # Process and measurement noise
        self.q_accel: float = q_accel
        self.R: np.ndarray = np.eye(2, dtype=np.float32) * r_meas
        self.t_last: float = time.time()

    def update(self, t: float, z_x: float, z_y: float) -> Tuple[float, float]:
        dt = t - self.t_last
        self.t_last = t
        if dt <= 0:
            dt = 1e-3

        # State transition F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        # Process noise covariance Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = np.array([
            [dt4, 0, dt3, 0],
            [0, dt4, 0, dt3],
            [dt3, 0, dt2, 0],
            [0, dt3, 0, dt2]
        ], dtype=np.float32) * self.q_accel

        # Predict
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

        # Measurement update
        z = np.array([z_x, z_y], dtype=np.float32)
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # Update
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4, dtype=np.float32) - K.dot(self.H)).dot(self.P)

        # Extract state estimates
        pos_x = float(self.x[0])
        pos_y = float(self.x[1])
        vx = float(self.x[2])
        vy = float(self.x[3])

        # Calculate velocity magnitude
        velocity_mag = math.sqrt(vx * vx + vy * vy)
        
        # Prevent drift when hand is stationary
        if velocity_mag < config.STATIONARY_VELOCITY_THRESHOLD:
            # Dampen state velocity to 0.0 to prevent drift accumulation
            self.x[2] = 0.0
            self.x[3] = 0.0
            vx = 0.0
            vy = 0.0
            pred_x = pos_x
            pred_y = pos_y
        else:
            # Linear motion prediction based on estimated velocity
            pred_x = pos_x + vx * config.LATENCY_COMPENSATION_SEC
            pred_y = pos_y + vy * config.LATENCY_COMPENSATION_SEC

        # Clamp prediction output to screen boundaries
        pred_x = max(0.0, min(float(config.SCREEN_WIDTH), pred_x))
        pred_y = max(0.0, min(float(config.SCREEN_HEIGHT), pred_y))

        return pred_x, pred_y


class MouseController:
    def __init__(self) -> None:
        # Smoothed screen coordinates
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        
        # Previous output coordinates for dead zone logic
        self.prev_output_x: Optional[float] = None
        self.prev_output_y: Optional[float] = None
        
        # Coordinates mapping states for acceleration
        self.prev_mapped_x: Optional[float] = None
        self.prev_mapped_y: Optional[float] = None
        self.prev_transform_time: Optional[float] = None
        self.cursor_target_x: float = 0.0
        self.cursor_target_y: float = 0.0

        # One Euro Filter instances
        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None

        # Kalman Filter instance
        self.kalman_filter: Optional[KalmanFilter2D] = None

        # Cooldowns and states
        self.click_cooldown_until: float = 0.0
        self.is_dragging: bool = False
        
        # Scroll tracking
        self.is_scrolling: bool = False
        self.last_scroll_x: Optional[float] = None
        self.last_scroll_y: Optional[float] = None
        self.scroll_vel_x: float = 0.0
        self.scroll_vel_y: float = 0.0

        # Offset to prevent cursor jump on Fist Drag transition
        self.fist_offset_x: float = 0.0
        self.fist_offset_y: float = 0.0
        self.was_fist: bool = False

        # Active Pause State (toggled by pause gesture)
        self.is_paused: bool = False
        self.last_pause_toggle_time: float = 0.0

        # Action feedback for HUD overlays
        self.last_click_event: Optional[str] = None
        self.last_click_time: float = 0.0

        # Landmark smoothing filters
        self.filtered_lm_x: Optional[float] = None
        self.filtered_lm_y: Optional[float] = None

        # High-precision scroll accumulator
        self.scroll_accumulator: float = 0.0

    def transform_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates (x, y) to screen pixels proportionally
        using the full camera frame.
        """
        mapped_x = max(0.0, min(1.0, x))
        mapped_y = max(0.0, min(1.0, y))
        
        target_x = mapped_x * config.SCREEN_WIDTH
        target_y = mapped_y * config.SCREEN_HEIGHT
        
        return int(target_x), int(target_y)

    def smooth_coordinates(self, target_x: float, target_y: float) -> Tuple[int, int]:
        """Applies configured smoothing algorithm and stationary dead zones."""
        current_time = time.time()
        
        if config.SMOOTHING_METHOD == "one_euro":
            if self.filter_x is None or self.filter_y is None:
                self.filter_x = OneEuroFilter(current_time, target_x,
                                             min_cutoff=config.ONE_EURO_FC_MIN,
                                             beta=config.ONE_EURO_BETA,
                                             d_cutoff=config.ONE_EURO_FC_D)
                self.filter_y = OneEuroFilter(current_time, target_y,
                                             min_cutoff=config.ONE_EURO_FC_MIN,
                                             beta=config.ONE_EURO_BETA,
                                             d_cutoff=config.ONE_EURO_FC_D)
                self.smooth_x = target_x
                self.smooth_y = target_y
            else:
                self.smooth_x = self.filter_x(current_time, target_x)
                self.smooth_y = self.filter_y(current_time, target_y)
                
        elif config.SMOOTHING_METHOD == "kalman":
            if self.kalman_filter is None:
                self.kalman_filter = KalmanFilter2D(target_x, target_y,
                                                    q_accel=config.KALMAN_Q_ACCEL,
                                                    r_meas=config.KALMAN_R_MEAS)
                self.smooth_x = target_x
                self.smooth_y = target_y
            else:
                self.smooth_x, self.smooth_y = self.kalman_filter.update(current_time, target_x, target_y)
                
        else:  # Fallback to EMA
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x = target_x
                self.smooth_y = target_y
            else:
                alpha = config.SMOOTHING_ALPHA
                self.smooth_x = alpha * target_x + (1 - alpha) * self.smooth_x
                self.smooth_y = alpha * target_y + (1 - alpha) * self.smooth_y

        # Deadzone logic to completely eliminate micro-jitter when stationary
        if self.prev_output_x is not None and self.prev_output_y is not None:
            dx = self.smooth_x - self.prev_output_x
            dy = self.smooth_y - self.prev_output_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < config.DEADZONE_PIXELS:
                self.smooth_x = self.prev_output_x
                self.smooth_y = self.prev_output_y

        self.prev_output_x = self.smooth_x
        self.prev_output_y = self.smooth_y
        
        return int(self.smooth_x), int(self.smooth_y)

    def reset_tracking(self) -> None:
        """Resets filters and coordinates mapping states to avoid jumps when hand is lost/found."""
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.is_dragging = False
            print("[Controller] Hand lost - releasing drag (mouseUp)")
        self.is_scrolling = False
        self.last_scroll_x = None
        self.last_scroll_y = None
        self.smooth_x = None
        self.smooth_y = None
        self.prev_output_x = None
        self.prev_output_y = None
        
        # Reset relative cursor mapping states
        self.prev_mapped_x = None
        self.prev_mapped_y = None
        self.prev_transform_time = None
        self.cursor_target_x = 0.0
        self.cursor_target_y = 0.0
        
        self.filter_x = None
        self.filter_y = None
        self.kalman_filter = None
        
        # Reset landmark filters and scroll accumulator
        self.filtered_lm_x = None
        self.filtered_lm_y = None
        self.scroll_accumulator = 0.0

    def execute_action(self, gesture: str, landmarks: Optional[Any], hand_label: Optional[str]) -> Optional[Tuple[int, int]]:
        """
        Translates gestures (Move Cursor, Left Click, Right Click, Scroll Mode, Pause)
        into system mouse actions.
        Returns:
            cursor_pos: tuple (x, y) representing current cursor position, or None if not moving/active.
        """
        current_time = time.time()

        # Reset states if no hand is detected
        if gesture == "No Hand" or landmarks is None:
            self.reset_tracking()
            return None

        # Handle Pause Cursor (Open Palm)
        if gesture == "Pause Cursor":
            self.is_paused = True
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self.is_dragging = False
                print("[Controller] Pause gesture - releasing drag (mouseUp)")
            # Return current cursor position to prevent movement while paused
            return (self.prev_output_x, self.prev_output_y) if self.prev_output_x is not None else None
        else:
            self.is_paused = False

        lms = landmarks.landmark

        # Track Index Finger Tip (Landmark 8) for cursor positioning
        tracking_lm = lms[8]

        # Adaptive landmark input smoothing to eliminate micro-jitter
        if self.filtered_lm_x is None or self.filtered_lm_y is None:
            self.filtered_lm_x = tracking_lm.x
            self.filtered_lm_y = tracking_lm.y
        else:
            dlm_x = tracking_lm.x - self.filtered_lm_x
            dlm_y = tracking_lm.y - self.filtered_lm_y
            dist_lm = math.sqrt(dlm_x*dlm_x + dlm_y*dlm_y)
            
            # Map distance to dynamic EMA smoothing factor:
            # Slow movements (small dist) are smoothed heavily, fast movements (large dist) respond instantly.
            min_alpha = 0.03
            max_alpha = 0.95
            alpha = min_alpha + (max_alpha - min_alpha) * min(1.0, dist_lm / 0.012)
            
            self.filtered_lm_x = alpha * tracking_lm.x + (1.0 - alpha) * self.filtered_lm_x
            self.filtered_lm_y = alpha * tracking_lm.y + (1.0 - alpha) * self.filtered_lm_y

        # Calculate relative movement and adaptive acceleration using filtered landmark positions
        if self.prev_mapped_x is not None and self.prev_transform_time is not None:
            dx_f = self.filtered_lm_x - self.prev_mapped_x
            dy_f = self.filtered_lm_y - self.prev_mapped_y
            dt = current_time - self.prev_transform_time
            if dt <= 0:
                dt = 1e-3
            
            # Index tip speed in normalized coordinates per second
            speed = math.sqrt(dx_f*dx_f + dy_f*dy_f) / dt
            
            # Adaptive acceleration multiplier
            multiplier = config.SENSITIVITY * (1.0 + config.ACCELERATION_FACTOR * speed)
            
            # Scale displacement to screen space
            dx_s = dx_f * config.SCREEN_WIDTH * multiplier
            dy_s = dy_f * config.SCREEN_HEIGHT * multiplier
            
            # Accumulate target coordinates
            self.cursor_target_x += dx_s
            self.cursor_target_y += dy_s
            
            # Clamp to screen boundary
            self.cursor_target_x = max(0.0, min(float(config.SCREEN_WIDTH), self.cursor_target_x))
            self.cursor_target_y = max(0.0, min(float(config.SCREEN_HEIGHT), self.cursor_target_y))
        else:
            # First time hand detected: anchor target to current physical cursor position
            current_px, current_py = pyautogui.position()
            self.cursor_target_x = float(current_px)
            self.cursor_target_y = float(current_py)
            self.smooth_x = float(current_px)
            self.smooth_y = float(current_py)
            
        self.prev_mapped_x = self.filtered_lm_x
        self.prev_mapped_y = self.filtered_lm_y
        self.prev_transform_time = current_time

        # Smooth target coordinates using configured method (Kalman, One Euro, EMA)
        smooth_x, smooth_y = self.smooth_coordinates(self.cursor_target_x, self.cursor_target_y)
        
        # Strictly clamp smoothed coordinates to physical screen boundaries to avoid pyautogui errors
        smooth_x = max(0, min(config.SCREEN_WIDTH - 1, int(smooth_x)))
        smooth_y = max(0, min(config.SCREEN_HEIGHT - 1, int(smooth_y)))

        # Handle Scroll Mode
        if gesture == "Scroll Mode":
            if not self.is_scrolling:
                self.is_scrolling = True
                # Set initial anchor point for displacement
                self.last_scroll_y = smooth_y
                self.scroll_accumulator = 0.0
            else:
                # Continuous scroll based on relative delta movement
                dy = smooth_y - self.last_scroll_y
                self.last_scroll_y = smooth_y
                
                # Accumulate fractional scroll steps (negative dy is up, positive dy is down)
                # Scroll speed is directly scaled by scroll sensitivity
                self.scroll_accumulator += -dy * (config.SCROLL_SENSITIVITY / 8.0)
                
                clicks = int(self.scroll_accumulator)
                if clicks != 0:
                    try:
                        pyautogui.scroll(clicks)
                    except Exception as e:
                        print(f"[Controller] Scroll error: {e}")
                    self.scroll_accumulator -= clicks
            # Freeze cursor position during scrolling to prevent runaway/jitter behavior
            return (self.prev_output_x, self.prev_output_y) if self.prev_output_x is not None else (smooth_x, smooth_y)
        else:
            self.is_scrolling = False
            self.last_scroll_y = None

        # Move the physical cursor (if not scrolling and not paused)
        try:
            pyautogui.moveTo(smooth_x, smooth_y)
        except pyautogui.FailSafeException:
            raise
        except Exception as e:
            print(f"[Controller] Move cursor error: {e}")

        # Click action execution with cooldown check
        if current_time > self.click_cooldown_until:
            if gesture == "Left Click":
                try:
                    pyautogui.click()
                except Exception as e:
                    print(f"[Controller] Left Click error: {e}")
                self.last_click_event = "left"
                self.last_click_time = current_time
                self.click_cooldown_until = current_time + config.CLICK_COOLDOWN_SEC
                print(f"[Controller] Executed Left Click at ({smooth_x}, {smooth_y})")
            elif gesture == "Right Click":
                try:
                    pyautogui.rightClick()
                except Exception as e:
                    print(f"[Controller] Right Click error: {e}")
                self.last_click_event = "right"
                self.last_click_time = current_time
                self.click_cooldown_until = current_time + config.CLICK_COOLDOWN_SEC
                print(f"[Controller] Executed Right Click at ({smooth_x}, {smooth_y})")

        return smooth_x, smooth_y
