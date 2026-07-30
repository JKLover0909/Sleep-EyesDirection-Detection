# FQC Sleep & Eye Direction Detection

Demo giám sát thao tác viên tại công đoạn **FQC**: phát hiện **ngủ gật / mệt** và **hướng nhìn mắt** (có đang nhìn màn hình kiểm tra hay không).

> Phạm vi demo **không** gồm bài toán đặt bo NG đúng vị trí.

## Tính năng

| Module | Cách hoạt động |
|---|---|
| **Sleep / nodding off** | Eye Aspect Ratio (EAR) liên tục thấp + rolling **PERCLOS** + proxy đầu gật (pitch) |
| **Eye direction** | Vị trí iris trong hốc mắt → Left / Right / Up / Down / Center |
| **Looking at screen** | Gaze ≈ `Center` trong khoảng thời gian đủ dài |
| **Dashboard** | MJPEG stream, metrics realtime, snapshot cảnh báo |

## Nguồn tham khảo đã mang về

| Nguồn | Vai trò |
|---|---|
| [e-candeloro/Driver-State-Detection](https://github.com/e-candeloro/Driver-State-Detection) | EAR, PERCLOS, timer attention (MIT) — video demo đã copy vào `data/sample_videos/` |
| [MohamedASAK/Gaze-Detection](https://github.com/MohamedASAK/Gaze-Detection) | Iris horizontal/vertical ratios → hướng nhìn |
| MediaPipe Face Mesh (478 landmarks + iris) | Face / eye / iris tracking realtime trên CPU |

### Dataset liên quan (chưa bắt buộc cho demo realtime)

| Dataset | Nội dung | Link |
|---|---|---|
| **YawDD** | Video ngáp / mệt khi lái | https://www.site.uottawa.ca/~shervin/yawdd/ |
| **MRL Eye Dataset** | Ảnh mắt mở/đóng | http://mrl.cs.vsb.cz/eyedataset |
| **UTA-RLDD** | Real-life drowsiness video | https://sites.google.com/view/utarldd/home |
| **NTHU Drowsy Driver** | Video mệt mỏi đa điều kiện | https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |

Demo hiện tại **không cần train model** — chạy heuristic landmark. Dataset trên dùng nếu muốn fine-tune CNN mắt mở/đóng sau này.

## Cấu trúc

```
backend/
  engine.py      # MediaPipe + EAR + gaze + alerts
  main.py        # FastAPI (port 8010)
  database.py    # SQLite events
  static/        # Dashboard
data/
  sample_videos/ # Video demo
  snapshots/     # Ảnh khi có alert
references/      # Ghi chú nguồn
third_party/     # LICENSE tham chiếu
```

## Chạy demo (tạm trong terminal)

```bash
cd /home/jkl/Code/Sleep-EyesDirection-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python main.py
```

## Chạy nền (tắt Cursor/SSH vẫn live)

```bash
# start (tmux + auto-restart nếu crash)
./scripts/start_detached.sh

# kiểm tra
./scripts/status.sh

# dừng
./scripts/stop.sh
```

Session tmux: `fqc-sleep-eyes` · log: `data/server.log`  
Dashboard: **http://\<server-ip\>:8010** (vd. `http://172.100.0.6:8010`)

- Chọn video mẫu `driver_demo.mp4` hoặc **Webcam**
- Overlay: landmarks / gaze arrow / HUD
- Panel phải: cảnh báo Sleep / LookingAway / Tired + snapshot

## Ngưỡng mặc định (chỉnh trong `engine.py`)

- `ear_thresh = 0.21` — mắt đóng
- `ear_time_thresh = 2.0s` — ngủ gật
- `gaze_away_time_thresh = 2.0s` — không nhìn màn hình
- `perclos_thresh = 0.25` trên cửa sổ 30s — mệt

## API nhanh

- `GET /video_feed` — MJPEG
- `GET /api/stats` — metrics live
- `GET /api/events` — danh sách cảnh báo
- `POST /api/source` — `{"source":"driver_demo.mp4"}` hoặc `"webcam"`
- `POST /api/overlays` — bật/tắt overlay

## Ghi chú triển khai FQC

- Camera nên nhìn **chính diện / hơi nghiêng** khuôn mặt thao tác viên.
- “Nhìn màn hình” đang map = gaze **Center**. Khi lắp thực tế có thể calibrate ROI màn hình (đổi ngưỡng `avg_h` / `avg_v`).
- CPU-only; không cần GPU cho MediaPipe Face Mesh.
