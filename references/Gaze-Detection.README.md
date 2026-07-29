# Gaze Detection

This project detects and visualizes the direction of a person's gaze in real-time using a webcam.

---

## Features

- **Real-time gaze direction detection:** Classifies gaze as Left, Right, Up, Down, Center, or combinations (e.g., Up-Left).
- **Visual feedback:** Draws arrows from the iris centers indicating gaze direction, and overlays text on the video feed.
- **Smoothing:** Uses a short history of predictions to stabilize the output and reduce flicker.
- **Robust iris tracking:** Uses MediaPipe’s refined landmarks for accurate iris localization.

---

## How It Works

1. **Face & Iris Detection:**  
   MediaPipe FaceMesh detects facial landmarks, including iris positions.

2. **Gaze Estimation:**  
   The code calculates the relative position of each iris within its eye using landmark ratios.  
   - Horizontal and vertical ratios are computed for both eyes.
   - These ratios are classified into gaze directions (Left, Right, Up, Down, Center).

3. **Smoothing:**  
   A short history of gaze predictions is kept. The most frequent recent prediction is used for display.

4. **Visualization:**  
   - Arrows are drawn from each iris center in the direction of the detected gaze.
   - The current gaze direction is displayed as text.

---

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)

Install dependencies:

```sh
pip install opencv-python mediapipe
```

---

## Usage

Run the application:

```sh
python app.py
```

- A window will open showing the webcam feed.
- Gaze direction is indicated by arrows and text.
- Press `ESC` to exit.

---

## File Structure

```
Gaze-Detection/
├── app.py       # Main application code
├── README.md    # Project documentation
```

---

## Customization

- **Smoothing window:**  
  Change `gaze_history = deque(maxlen=10)` for more/less smoothing.
- **Arrow length:**  
  Adjust `length` in `draw_arrow()` for longer/shorter arrows.
- **Gaze thresholds:**  
  Modify values in `get_gaze_direction()` for sensitivity tuning.