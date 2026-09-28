# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## What this repo is

Demo giám sát thao tác viên tại công đoạn FQC: phát hiện ngủ gật (EAR + PERCLOS) và hướng nhìn mắt (gaze qua iris tracking, MediaPipe Face Mesh), có dashboard FastAPI + MJPEG stream. **Không** xử lý bài toán đặt bo NG đúng vị trí.

## Layout

```
backend/engine.py     # MediaPipe + EAR + gaze + logic cảnh báo — ngưỡng mặc định ở đây
backend/main.py       # FastAPI app, port 8010
backend/database.py   # SQLite lưu events
models/face_landmarker.task  # MediaPipe model (~3.6MB, đã commit — không phải rác)
data/sample_videos/   # video demo (đã commit chủ đích, xem README)
third_party/          # LICENSE của code tham khảo (Driver-State-Detection, MIT)
references/           # ghi chú nguồn tham khảo (SOURCES.md)
```

## Ràng buộc

- Repo này **mang code/ý tưởng từ 2 nguồn MIT khác** (`e-candeloro/Driver-State-Detection`, `MohamedASAK/Gaze-Detection`) — giữ nguyên `third_party/Driver-State-Detection.LICENSE` và `references/SOURCES.md`, không xóa khi dọn dẹp.
- Đổi ngưỡng detection (EAR, PERCLOS, gaze timeout) ở `backend/engine.py`, không hard-code lại ở nơi khác.
- CPU-only, không cần GPU — không tự thêm dependency GPU-only mà không hỏi trước.
- `scripts/start_detached.sh`/`stop.sh`/`status.sh` dùng tmux để chạy nền — không sửa cách quản lý process này sang thứ khác (systemd, docker...) mà không hỏi, vì `scripts/fqc-sleep-eyes.service` có thể đang được dùng song song.

## Kiểm thử an toàn

```bash
python -c "import ast; ast.parse(open('backend/engine.py').read())"
```
Không chạy `backend/main.py` thật (cần webcam/video + MediaPipe runtime, không có trong môi trường agent).
