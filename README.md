# Hand Gesture Controller

Ứng dụng tương tác thời gian thực bằng cử chỉ tay, xây dựng với **Python**, **OpenCV**
và **MediaPipe**. Chương trình lấy hình ảnh từ webcam, phát hiện 21 landmark trên
bàn tay, nhận diện cử chỉ tĩnh/chuyển động và cho phép người dùng thao tác với các
hình học trực tiếp trên khung hình mà không cần chuột.

> Đây là dự án cá nhân về Computer Vision và Human–Computer Interaction. Mục tiêu
> của dự án là minh họa một pipeline thị giác máy tính hoàn chỉnh: camera input,
> landmark detection, gesture state machine, interaction logic và real-time rendering.

## Mục lục

- [Tính năng](#tính-năng)
- [Cách sử dụng cử chỉ](#cách-sử-dụng-cử-chỉ)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Chạy kiểm thử](#chạy-kiểm-thử)
- [Đóng gói ứng dụng Windows](#đóng-gói-ứng-dụng-windows)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Thuật toán chính](#thuật-toán-chính)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Giới hạn hiện tại](#giới-hạn-hiện-tại)
- [Định hướng phát triển](#định-hướng-phát-triển)

## Tính năng

- Đọc và xử lý video webcam theo thời gian thực.
- Phát hiện tối đa hai bàn tay bằng MediaPipe Hands.
- Hiển thị landmark và skeleton của bàn tay trên frame.
- Xác định trạng thái mở/đóng của năm ngón tay.
- Hiển thị icon trạng thái ngón tay riêng cho tay trái và tay phải.
- Nhận diện cử chỉ tĩnh và cử chỉ có chuyển động.
- Bật/tắt chế độ tương tác bằng chuỗi cử chỉ.
- Tạo bốn loại hình: chữ nhật, hình tròn, tam giác và ngôi sao.
- Chọn, kéo thả, đổi màu và xóa hình bằng cử chỉ.
- Giới hạn vật thể trong khung hình và xử lý đúng thứ tự khi các hình chồng nhau.
- Hiển thị FPS để theo dõi hiệu năng xử lý.
- Hỗ trợ chọn camera qua tham số dòng lệnh.
- Có cấu hình PyInstaller để tạo file thực thi trên Windows.

## Cách sử dụng cử chỉ

### Cử chỉ tĩnh

| Cử chỉ | Ý nghĩa trong ứng dụng |
|---|---|
| `Fist` | Nắm tay |
| `Stop` | Xòe năm ngón; xóa hình tại vị trí con trỏ |
| `Peace` | Giơ ngón trỏ và ngón giữa |
| `OK` | Ngón cái chạm ngón trỏ, các ngón còn lại mở |
| `Select` | Chụm ngón cái và ngón trỏ để chọn/kéo hình hoặc bấm menu |
| `Options` | Đổi màu hình tại vị trí con trỏ |
| `Thumbs Up` | Ngón cái hướng lên |
| `Thumbs Down` | Ngón cái hướng xuống |

### Cử chỉ chuyển động

| Cử chỉ | Cách nhận diện / tác dụng |
|---|---|
| `Move Left/Right/Up/Down` | Xòe bàn tay và di chuyển theo hướng tương ứng |
| `Wave` | Xòe bàn tay và đổi hướng trái–phải liên tiếp |
| `SOS` | Chuyển nhanh từ bốn ngón mở sang nắm tay |
| `On/Off` | Thực hiện chuỗi cử chỉ bật/tắt để hiện hoặc ẩn chế độ tương tác |
| `Still` | Không có chuyển động đủ lớn |

Con trỏ tương tác nằm tại trung điểm giữa đầu ngón cái và đầu ngón trỏ. Do hệ
thống hiện dùng heuristic hình học, người dùng nên giữ bàn tay hướng tương đối
thẳng về phía camera và tránh để các ngón che khuất nhau.

### Thao tác với hình

1. Thực hiện cử chỉ `On/Off` để bật giao diện tương tác.
2. Dùng `Select` tại nút **MENU** ở góc dưới bên trái.
3. Dùng `Select` trên một hình mẫu để tạo hình mới.
4. Giữ `Select` trên hình và di chuyển tay để kéo hình.
5. Dùng `Options` trên hình để thay đổi màu.
6. Dùng `Stop` trên hình để xóa hình.
7. Nhấn phím `q` để thoát ứng dụng.

## Kiến trúc hệ thống

Luồng xử lý chính:

```text
Webcam
  │
  ▼
OpenCV frame capture và horizontal flip
  │
  ▼
MediaPipe Hands ──► 21 landmarks / bàn tay
  │
  ├──► FingerNumber ──► mã trạng thái 5 ngón + icon
  │
  └──► GestureDetector
          ├──► static gesture classifier
          └──► temporal motion state machine
                         │
                         ▼
              DraggableObjectManager
                         │
                         ▼
            Shape menu + object rendering
                         │
                         ▼
                  OpenCV display
```

Ứng dụng tách phần phát hiện bàn tay, nhận diện gesture, quản lý object và vòng
lặp camera thành các module riêng. `Main.py` chỉ điều phối các thành phần và có
`main()` guard, vì vậy có thể import module khi viết test mà không tự động mở camera.

## Yêu cầu hệ thống

- Windows, Linux hoặc macOS có webcam.
- Python 3.10 hoặc Python 3.11 được khuyến nghị.
- Webcam tích hợp hoặc webcam USB.
- Môi trường đủ sáng để landmark detection ổn định.

Các thư viện chính:

- OpenCV: đọc camera, xử lý và render frame.
- MediaPipe: phát hiện bàn tay và 21 landmark.
- NumPy: biểu diễn tọa độ polygon.
- PyInstaller: đóng gói ứng dụng Windows.

## Cài đặt

Clone repository và chuyển vào thư mục dự án:

```powershell
git clone <repository-url>
cd <repository-folder>
```

Tạo virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Trên Command Prompt có thể kích hoạt bằng:

```bat
.venv\Scripts\activate.bat
```

Cài dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

Chạy với camera mặc định:

```powershell
python Main.py
```

Chọn camera cụ thể:

```powershell
python Main.py --camera 1
```

Ứng dụng sẽ thử camera được chỉ định trước, sau đó fallback sang camera `0` hoặc
`1` nếu camera đó không mở được.

## Chạy kiểm thử

Dự án sử dụng `unittest` có sẵn trong Python:

```powershell
python -m unittest discover -s tests -v
```

Bộ test hiện kiểm tra các hành vi quan trọng:

- Tay đứng yên phải trả về `Still`.
- Timer nhận diện `Wave` được khởi tạo đúng lúc.
- Reset xóa trạng thái temporal cũ.
- Vật thể không thể bị kéo ra ngoài frame.
- Mất dấu bàn tay phải kết thúc trạng thái kéo.

Kiểm tra cú pháp toàn bộ module:

```powershell
python -m py_compile Main.py Hand_Detector.py Finger_Number.py GestureDetector.py DraggableObject.py
```

## Đóng gói ứng dụng Windows

Repository có file cấu hình `Hand Gesture Controller.spec`. Để tạo executable:

```powershell
pyinstaller --clean "Hand Gesture Controller.spec"
```

File kết quả được tạo trong:

```text
dist/Hand Gesture Controller.exe
```

Các thư mục `build/` và `dist/` là sản phẩm sinh tự động, không nên commit vào
Git repository. Khi phát hành phiên bản mới, nên đính kèm executable trong phần
GitHub Releases.

## Cấu trúc thư mục

```text
.
├── Main.py                       # Application lifecycle và camera loop
├── Hand_Detector.py              # MediaPipe Hands wrapper
├── GestureDetector.py            # Static classifier và motion state machine
├── Finger_Number.py              # Mã trạng thái ngón tay và image cache
├── DraggableObject.py            # Shapes, hit testing, drag/drop và menu
├── Image/
│   ├── hand_left/                # Icon trạng thái tay trái
│   └── hand_right/               # Icon trạng thái tay phải
├── tests/
│   └── test_logic.py             # Unit tests cho gesture và object logic
├── requirements.txt              # Python dependencies
├── Hand Gesture Controller.spec  # PyInstaller configuration
├── .gitignore
└── README.md
```

## Thuật toán chính

### 1. Phát hiện bàn tay

Mỗi frame BGR từ OpenCV được chuyển sang RGB và đưa vào MediaPipe Hands. Kết quả
bao gồm 21 landmark chuẩn hóa theo chiều rộng và chiều cao frame. Skeleton được
vẽ lại để người dùng quan sát chất lượng tracking.

### 2. Xác định trạng thái ngón tay

Phiên bản hiện tại so sánh khoảng cách từ đầu ngón và khớp ngón tới điểm tham
chiếu. Năm trạng thái nhị phân được ghép thành mã như `01001`, sau đó dùng để tải
icon tương ứng. Ảnh đã đọc được cache để tránh truy cập ổ đĩa ở mỗi frame.

### 3. Nhận diện cử chỉ tĩnh

Cử chỉ tĩnh được phân loại bằng số ngón đang mở, vị trí tương đối theo trục `y`
và khoảng cách giữa các đầu ngón. Ví dụ, `Select` và `OK` sử dụng khoảng cách giữa
đầu ngón cái và đầu ngón trỏ.

### 4. Nhận diện chuyển động

Tâm bàn tay được ước lượng từ cổ tay và các đầu ngón. Vector dịch chuyển giữa hai
frame được dùng để phân loại bốn hướng. Các gesture theo chuỗi như `Wave`, `SOS`
và `On/Off` được quản lý bằng state machine có timeout, thay vì chỉ xét một frame.

### 5. Tương tác với vật thể

Con trỏ là trung điểm giữa đầu ngón cái và ngón trỏ. Mỗi shape tự cài đặt hit
testing phù hợp: bounding box cho hình chữ nhật, khoảng cách tâm cho hình tròn và
polygon test cho tam giác/ngôi sao. Manager chịu trách nhiệm z-order, drag state,
giới hạn tọa độ và các thao tác xóa/đổi màu.

## Xử lý sự cố

### Không mở được camera

- Đóng ứng dụng khác đang sử dụng webcam.
- Kiểm tra quyền truy cập camera của hệ điều hành.
- Thử `python Main.py --camera 1` hoặc một index khác.
- Kiểm tra camera với một ứng dụng webcam thông thường trước.

### `ModuleNotFoundError`

Đảm bảo virtual environment đã được kích hoạt và chạy lại:

```powershell
python -m pip install -r requirements.txt
```

### Gesture bị nhấp nháy hoặc nhận diện sai

- Tăng ánh sáng phía trước bàn tay.
- Giữ toàn bộ bàn tay trong frame.
- Tránh nền có màu hoặc texture quá giống bàn tay.
- Giữ lòng bàn tay tương đối hướng về camera.
- Di chuyển chậm và rõ ràng đối với gesture chuyển động.

### FPS thấp

- Đóng các ứng dụng camera hoặc ứng dụng nặng khác.
- Giảm độ phân giải camera trong `HandGestureApp`.
- Tắt việc vẽ landmark khi benchmark nếu chỉ cần đo inference.

## Giới hạn hiện tại

- Gesture classifier đang dựa trên heuristic, chưa phải mô hình học máy được huấn luyện.
- Threshold chưa được chuẩn hóa hoàn toàn theo kích thước và góc xoay bàn tay.
- Chưa có bộ dữ liệu video có nhãn để đo precision, recall hoặc confusion matrix.
- Tay điều khiển chính hiện là bàn tay đầu tiên MediaPipe trả về; thứ tự có thể đổi khi hai tay giao nhau.
- Chưa có smoothing nâng cao cho landmark và gesture output.
- Chưa có giao diện cấu hình trực quan cho threshold, camera và handedness.

## Định hướng phát triển

- Chuẩn hóa khoảng cách theo kích thước lòng bàn tay.
- Dùng góc khớp MCP–PIP–DIP–TIP thay cho một số luật khoảng cách đơn giản.
- Thêm temporal smoothing, debounce và confidence score.
- Giữ ổn định tay điều khiển dựa trên handedness hoặc tracking ID.
- Xây dựng dataset video nhỏ và công cụ gán nhãn gesture.
- Báo cáo FPS, latency, accuracy và confusion matrix.
- Thêm integration test với video mẫu thay vì phụ thuộc hoàn toàn vào webcam.
- Thêm GitHub Actions để tự động chạy lint và unit test.
- Tạo GIF/video demo và phát hành executable qua GitHub Releases.
- Thêm tính năng lưu/khôi phục bố cục các hình.

## Gợi ý mô tả trong CV

**Real-time Hand Gesture Interaction System — Python, OpenCV, MediaPipe**

> Developed a real-time vision-based interaction system using MediaPipe hand
> landmarks, geometric gesture classification, and temporal state machines.
> Implemented gesture-controlled shape creation, selection, dragging, deletion,
> color changes, multi-hand visualization, unit tests, and Windows packaging.

## License

Dự án hiện chưa khai báo license. Trước khi public repository, nên bổ sung một
license phù hợp, ví dụ MIT License, nếu bạn muốn người khác được phép sử dụng và
phát triển mã nguồn.
