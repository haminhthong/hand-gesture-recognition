import math
import time


class GestureDetector:
    def __init__(self):
        self.STATIC_GESTURES = ["Fist", "OK", "Thumbs Up", "Thumbs Down", "Peace", "Stop", "Select", "Options"]
        self.MOTION_GESTURES = ["SOS", "Wave", "Move Left", "Move Right", "Move Up", "Move Down", "Still", "On/Off"]
        self.GESTURES = self.STATIC_GESTURES + self.MOTION_GESTURES
        self.COLORS = {
            # Cử chỉ tĩnh
            "Fist": (0, 0, 255),         # Đỏ
            "OK": (0, 255, 0),           # Xanh lá
            "Thumbs Up": (255, 255, 0),  # Vàng
            "Thumbs Down": (255, 128, 0),# Cam đậm
            "Stop": (0, 165, 255),       # Cam
            "Peace": (255, 0, 255),      # Tím
            "Select": (128, 0, 128),     # Màu tím đậm
            "Options": (0, 128, 128),    # Màu xanh đậm
            "Unknown": (255, 255, 255),  # Xám
            # Cử chỉ chuyển động
            "SOS": (0, 0, 255),          # Đỏ
            "Wave": (255, 255, 0),       # Vàng
            "Move Left": (255, 0, 255),  # Tím
            "Move Right": (0, 255, 255), # Xanh lơ
            "Move Up": (0, 255, 0),      # Xanh lá
            "Move Down": (255, 165, 0),  # Cam
            "Swipe Left": (255, 0, 255), # Tím
            "Swipe Right": (0, 255, 255),# Xanh lơ
            "Still": (255, 255, 255),    # Trắng
            "On/Off": (128, 128, 0),     # Màu vàng đậm
        }
        self.last_detect_gesture = ("Still", self.COLORS["Still"])
        self.gestureTracking = ""
        self.sTime = 0
        self.eTime = time.perf_counter()
        
        # Theo dõi cử chỉ vẫy tay.
        self.wave_direction = ""  # Hướng trái hoặc phải
        self.wave_count = 0
        # Theo dõi vị trí trung tâm để xác định hướng di chuyển
        self.prev_center = None
        self.move_threshold = 0.015  # Ngưỡng di chuyển tối thiểu giữa hai khung hình

    def reset(self):
        """Xóa trạng thái theo thời gian khi mất dấu bàn tay."""
        self.last_detect_gesture = ("Still", self.COLORS["Still"])
        self.gestureTracking = ""
        self.sTime = 0
        self.eTime = time.perf_counter()
        self.wave_direction = ""
        self.wave_count = 0
        self.prev_center = None
    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def get_hand_center(self, landmarks):
        lm = landmarks.landmark
        center_x = (lm[0].x + lm[4].x + lm[8].x + lm[12].x + lm[16].x + lm[20].x) / 6
        center_y = (lm[0].y + lm[4].y + lm[8].y + lm[12].y + lm[16].y + lm[20].y) / 6
        return center_x, center_y
    
    def get_move_direction(self, current_center, prev_center):
        if prev_center is None:
            return "Still"
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        
        # Tính khoảng cách di chuyển
        distance = math.hypot(dx, dy)
        if distance < self.move_threshold:
            return "Still"
        
        # Tính góc so với trục y (0 độ = lên trên, 90 độ = sang phải)
        angle = math.atan2(dx, -dy) * 180 / math.pi  # -dy vì y tăng xuống dưới
        
        # Xác định hướng dựa trên góc
        if -45 <= angle <= 45:
            return "Move Up"  # Di chuyển lên
        elif 45 < angle <= 135:
            return "Move Right"  # Di chuyển sang phải
        elif angle > 135 or angle < -135:
            return "Move Down"  # Di chuyển xuống
        else:  # -135 <= angle < -45
            return "Move Left"  # Di chuyển sang trái
    
    def is_finger_up(self, landmarks, tip_id, mcp_id, wrist_id=0):
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(landmarks[mcp_id], landmarks[wrist_id])
    
    def is_thumb_up(self, landmarks, tip_id, mcp_id, wrist_id=17):
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(landmarks[mcp_id], landmarks[wrist_id])

    def detect_static_gesture(self, landmarks):
        if not landmarks:
            return "Unknown", self.COLORS["Unknown"]
        
        lm = landmarks.landmark
        wrist = lm[0]
        thumb_tip= lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        
        # Đếm số ngón tay "thẳng"
        fingers_up = 0
        if self.is_thumb_up(lm, 4, 3): fingers_up += 1
        if self.is_finger_up(lm, 8, 6): fingers_up += 1
        if self.is_finger_up(lm, 12, 10): fingers_up += 1
        if self.is_finger_up(lm, 16, 14): fingers_up += 1
        if self.is_finger_up(lm, 20, 18): fingers_up += 1
        
        gesture = "Unknown"
        if fingers_up == 0:
            gesture = "Fist"
        elif fingers_up == 5:
            gesture = "Stop"
        elif fingers_up == 2 and self.is_finger_up(lm, 8, 6) and self.is_finger_up(lm, 12, 10) and not self.is_finger_up(lm, 4, 3):
            gesture = "Peace"
        elif self.distance(thumb_tip, index_tip) < 0.06:
            if self.distance(middle_tip, index_tip)<0.06:
                gesture = "Options"
            elif self.distance(middle_tip, index_tip)>0.1 and fingers_up == 2:
                gesture = "Select"
            elif fingers_up >= 3:
                gesture = "OK"
        elif fingers_up == 1 and self.is_thumb_up(lm, 4, 2) and thumb_tip.y < wrist.y:
            gesture = "Thumbs Up"
        elif fingers_up == 1 and self.is_thumb_up(lm, 4, 2) and thumb_tip.y > wrist.y:
            gesture = "Thumbs Down"
        color = self.COLORS.get(gesture, self.COLORS["Unknown"])
        return gesture, color
    
    def detect_motion_gesture(self, landmarks):
        if not landmarks:
            return self.last_detect_gesture, self.gestureTracking
        
        lm = landmarks.landmark
        
        # Đếm số ngón tay "thẳng"
        fingers_up = 0
        if self.is_thumb_up(lm, 4, 3): fingers_up += 1
        if self.is_finger_up(lm, 8, 6): fingers_up += 1
        if self.is_finger_up(lm, 12, 10): fingers_up += 1
        if self.is_finger_up(lm, 16, 14): fingers_up += 1
        if self.is_finger_up(lm, 20, 18): fingers_up += 1
        
        # Mỗi khung hình bắt đầu ở trạng thái đứng yên để không giữ lại hướng cũ.
        gesture = "Still"
        
        if not self.is_thumb_up(lm, 4, 3) and fingers_up == 4:
            if self.gestureTracking != "Start SOS":
                self.sTime = time.perf_counter()
                self.gestureTracking = "Start SOS"
        elif fingers_up == 0 and self.gestureTracking == "Start SOS":
            self.eTime = time.perf_counter()
            dt = self.eTime - self.sTime
            if dt < 1.5:
                gesture = "SOS"
            self.gestureTracking = ""
        elif self.is_finger_up(lm, 8, 6) and self.is_finger_up(lm, 12, 10) and self.distance(lm[8], lm[4]) >= 0.1 and self.distance(lm[12], lm[4]) < 0.06 and not self.is_finger_up(lm, 16, 14) and not self.is_finger_up(lm, 20, 18) :
            self.gestureTracking = "Start onoff"
        elif self.is_finger_up(lm, 8, 6) and not self.is_finger_up(lm, 12, 10) and self.distance(lm[8], lm[4]) >= 0.1 and self.distance(lm[12], lm[4]) >= 0.06 and not self.is_finger_up(lm, 16, 14) and not self.is_finger_up(lm, 20, 18) and self.gestureTracking == "Start onoff":
            gesture = "On/Off"
            self.gestureTracking = ""

        elif fingers_up == 5:
            current_center = self.get_hand_center(landmarks)
            move_direction = self.get_move_direction(current_center, self.prev_center)
            wave_timed_out = False
            
            if self.gestureTracking != "Start Wave":
                self.gestureTracking = "Start Wave"
                self.wave_direction = "center"
                self.wave_count = 0
                self.sTime = time.perf_counter()
            else:  
                if move_direction == "Move Left" and self.wave_direction != "left":
                    self.wave_direction = "left"
                    self.wave_count += 1
                elif move_direction == "Move Right" and self.wave_direction != "right":
                    self.wave_direction = "right"
                    self.wave_count += 1
                if self.wave_count >= 3:
                    gesture = "Wave"
                self.eTime = time.perf_counter()
                wave_timed_out = self.eTime - self.sTime >= 2.0
                if wave_timed_out:
                    gesture = "Still"
                    self.gestureTracking = ""
                    self.wave_direction = "center"
                    self.wave_count = 0
                    self.prev_center = None
            if gesture != "Wave" and not wave_timed_out:
                gesture = move_direction
            if not wave_timed_out:
                self.prev_center = current_center
        elif self.gestureTracking == "Start Wave":
            gesture = "Still"
            self.gestureTracking = ""
            self.wave_direction = "center"
            self.wave_count = 0
            self.prev_center = None

        color = self.COLORS.get(gesture, self.COLORS["Still"])
        self.last_detect_gesture = gesture, color
        return self.last_detect_gesture, self.gestureTracking
    
    def detect_gesture(self, landmarks, mode="motion"):
        if not landmarks:
            if mode == "static":
                return "Unknown", self.COLORS["Unknown"]
            elif mode == "motion":
                return self.last_detect_gesture, self.gestureTracking
            else:  # Cả hai loại cử chỉ
                return {
                    "static": ("Unknown", self.COLORS["Unknown"]),
                    "motion": (self.last_detect_gesture, self.gestureTracking)
                }
        if mode == "static":
            return self.detect_static_gesture(landmarks)
        elif mode == "motion":
            return self.detect_motion_gesture(landmarks)
        else:  # Cả hai loại cử chỉ
            static_result = self.detect_static_gesture(landmarks)
            motion_result = self.detect_motion_gesture(landmarks)
            return {
                "static": static_result,
                "motion": motion_result
            }
