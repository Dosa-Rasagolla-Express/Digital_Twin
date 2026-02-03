# Vehicle Speed Detection System

A real-time vehicle detection and speed estimation system built with **Python**, **OpenCV**, and **YOLOv8**.

## 🚀 Features
- **Object Detection**: Uses the YOLOv8 Nano model (`yolov8n`) for fast and accurate vehicle detection.
- **Vehicle Tracking**: Tracks specific vehicle classes (cars, motorcycles, buses, trucks) across frames.
- **Speed Estimation**: Calculates approximate vehicle speed in km/h using pixel displacement and time difference.
- **Visualisation**: Draws bounding boxes and displays speed labels directly on the video feed.

## 🛠️ Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dosa-Rasagolla-Express/Vehicle_count-speed.git
   cd Vehicle_count-speed
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 weights (will usually download automatically on first run):
   - `yolov8n.pt`

## 🎬 Usage
1. Place your input video in the `videos/` folder (default is `videos/traffic.mp4`).
2. Run the script:
   ```bash
   python main.py
   ```
3. Press `Esc` to exit the video window.

## ⚙️ Configuration
- You can adjust the `PIXEL_TO_METER` constant in `main.py` to calibrate speed estimation for your specific camera angle and resolution.

---
Made with love 💛 by Raskhith Ganjimut and AI
# speed
# speed
# speed
# speed
# speed
