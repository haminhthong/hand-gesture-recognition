# Dataset card

Repo hiện chưa phân phối dataset gán nhãn. Thư mục này quy định schema và
protocol cần tuân thủ khi thu thập dữ liệu để tránh báo cáo kết quả sai lệch.

## Schema tối thiểu

Mỗi mẫu cần có: `subject_id`, `session_id`, `gesture`, `frame_index`,
`handedness`, 21 điểm mốc `x/y/z`, độ phân giải, thiết bị và điều kiện sáng.

Không lưu tên thật hoặc dữ liệu nhận dạng cá nhân trong repository công khai.

## Protocol đánh giá

- Chia train/validation/test theo `subject_id`, không chia ngẫu nhiên theo frame.
- Các frame cùng video hoặc session chỉ được xuất hiện trong một split.
- Mọi bước chuẩn hóa học từ dữ liệu phải chỉ fit trên train.
- Test cuối chỉ dùng một lần để báo cáo kết quả.
- Báo cáo Accuracy, Macro-F1, confusion matrix và latency theo từng người.

## Checklist trước khi công bố metric

- Ghi số người, số session, số mẫu mỗi lớp và tỷ lệ tay trái/phải.
- Kiểm tra mẫu trùng và leakage giữa các split.
- Ghi phiên bản camera, MediaPipe, seed và cấu hình mô hình.
- Phân tích lỗi theo ánh sáng, khoảng cách camera và người dùng.
