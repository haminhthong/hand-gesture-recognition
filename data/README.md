# Dataset Card & Data Collection Protocol

Thư mục này quy định cấu trúc (Schema) tệp dữ liệu CSV và quy trình (Protocol) thu thập, phân chia dataset để phục vụ huấn luyện và đánh giá các mô hình Machine Learning (KNN, SVM, Random Forest) một cách trung thực, tránh rò rỉ dữ liệu (Data Leakage).

## 📊 CSV Dataset Schema

Mỗi dòng dữ liệu CSV thu thập từ `tools/collect_landmarks.py` tuân thủ tiêu chuẩn cột sau:

```text
sample_id, subject_id, session_id, gesture, frame_index, timestamp_ms, handedness, camera_width, camera_height, device_id, lighting, x0, y0, z0, ..., x20, y20, z20
```

### Chi tiết ý nghĩa các trường:
- `sample_id` (str): Mã định danh duy nhất cho từng mẫu (UUID hoặc `sub001_s001_f0001`).
- `subject_id` (str): Mã định danh người tham gia thu thập (ví dụ: `subject_001`, `subject_002`). **Không dùng tên thật**.
- `session_id` (str): Mã phiên thu thập (ví dụ: `session_001`, `session_002`).
- `gesture` (str): Nhãn cử chỉ (ví dụ: `Fist`, `Stop`, `OK`, `Peace`, `Select`, `Unknown`).
- `frame_index` (int): Thứ tự khung hình trong phiên thu thập.
- `timestamp_ms` (float): Mốc thời gian thu thập (mili-giây).
- `handedness` (str): Bàn tay được nhận diện (`Left` hoặc `Right`).
- `camera_width` / `camera_height` (int): Độ phân giải khung hình webcam (ví dụ: `640x480` hoặc `1280x720`).
- `device_id` (str): Mã định danh thiết bị/camera (ví dụ: `cam_laptop`, `cam_usb_01`).
- `lighting` (str): Điều kiện ánh sáng (`normal`, `dim`, `bright`).
- `x0, y0, z0 ... x20, y20, z20` (float): Tọa độ 21 điểm mốc 3D trích xuất từ MediaPipe Hands.

---

## 🛡️ Protocol Đánh Giá & Chống Data Leakage

1. **Phân chia split theo người (`subject_id`)**:
   - Chia `train`, `validation`, `test` dựa trên `subject_id`, tuyệt đối **không** xáo trộn ngẫu nhiên (random shuffle) từng dòng dữ liệu frame.
   - Ví dụ:
     - `train_subjects`: `subject_001` $\to$ `subject_006`
     - `val_subjects`: `subject_007` $\to$ `subject_008`
     - `test_subjects`: `subject_009` $\to$ `subject_010`
   - Đảm bảo tính rời rạc tuyệt đối giữa các tập:
     $$\text{train\_subjects} \cap \text{val\_subjects} = \emptyset$$
     $$\text{train\_subjects} \cap \text{test\_subjects} = \emptyset$$
     $$\text{val\_subjects} \cap \text{test\_subjects} = \emptyset$$

2. **Cross-Validation**:
   - Khi dữ liệu ít người tham gia, sử dụng `GroupKFold` hoặc `LeaveOneGroupOut` với `groups = dataset['subject_id']`.

3. **Chuẩn hóa dữ liệu (Preprocessing)**:
   - Các thao tác fit thống kê (nếu có) phải thực hiện duy nhất trên tập `train`.
   - Lớp `LandmarkPreprocessor` (`src/hand_gesture_controller/preprocessing.py`) chuẩn hóa theo kích thước lòng bàn tay độc lập trên từng mẫu bàn tay (zero-parameter transformation), ngăn ngừa rò rỉ thông tin giữa các mẫu.

---

## 🔒 Quy Định Đồng Ý & Bảo Mật (Consent & Privacy)

- Giải thích mục đích thu thập dữ liệu cho người tham gia.
- Chỉ lưu trữ các chỉ số tọa độ số landmarks 3D; tuyệt đối **không** lưu trữ ảnh khuôn mặt hoặc video webcam chưa che danh tính.
- Dữ liệu CSV thô thu thập cục bộ phải nằm trong thư mục được cấu hình ẩn (`data/raw/` hoặc `data/private/`), không đưa thông tin nhận dạng cá nhân lên Git repository công khai.
