import math
import os

import cv2


class FingerNumber:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.hand_left_path = os.path.join(base_dir, "Image", "hand_left")
        self.hand_right_path = os.path.join(base_dir, "Image", "hand_right")
        
        # Lưu ảnh đã đọc để không phải truy cập ổ đĩa ở mỗi khung hình.
        self.left_hand_images = {}
        self.right_hand_images = {}
    
    @staticmethod
    def distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    def is_finger_up(self, landmarks, tip_id, mcp_id, wrist_id=0):
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(landmarks[mcp_id], landmarks[wrist_id])
    
    def is_thumb_up(self, landmarks, tip_id, mcp_id, wrist_id=17):
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(landmarks[mcp_id], landmarks[wrist_id])

    def detect_gesture(self, landmarks, hand_label="Right"):
        if not landmarks:
            return "00000", None
            
        lm = landmarks.landmark  # Danh sách 21 điểm mốc
        finger_code_value = 0
        if self.is_thumb_up(lm, 4, 3):
            finger_code_value += 1
        if self.is_finger_up(lm, 8, 6):
            finger_code_value += 10
        if self.is_finger_up(lm, 12, 10):
            finger_code_value += 100
        if self.is_finger_up(lm, 16, 14):
            finger_code_value += 1000
        if self.is_finger_up(lm, 20, 18):
            finger_code_value += 10000
        
        finger_code = f"{finger_code_value:05d}"  # Chuỗi năm chữ số
        
        if hand_label == "Left":
            img_path = os.path.join(self.hand_left_path, f"{finger_code}.png")
            cache = self.left_hand_images
        else:
            img_path = os.path.join(self.hand_right_path, f"{finger_code}.png")
            cache = self.right_hand_images
        
        # Chỉ đọc ảnh ở lần đầu gặp trạng thái ngón tay này.
        if finger_code not in cache:
            if os.path.exists(img_path):
                cache[finger_code] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            else:
                cache[finger_code] = None
        
        return finger_code, cache[finger_code]
        
        
