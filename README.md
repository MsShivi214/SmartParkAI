# Advanced AI Car Parking Live Management System

This project is a high-performance car parking space detection and monitoring system. It leverages computer vision and deep learning to identify parking occupancy, detect vehicles, and recognize license plates in real-time. The system features a modern web-based dashboard for live monitoring.

## 🚀 Key Features

- **Live Video Dashboard**: Real-time web interface with synchronized video stream and status table.
- **Hybrid Occupancy Detection**: Combines YOLOv8 vehicle detection with adaptive pixel-density analysis for maximum accuracy.
- **Temporal Consistency**: Majority-vote filtering across frames to eliminate status flickering and detection noise.
- **Spatially Ordered Indexing**: Automatic lane-wise sorting (Left-to-Right, Top-to-Bottom) for intuitive slot numbering.
- **License Plate Recognition (LPR)**: Integrated **EasyOCR** to automatically extract license plate numbers from occupied slots.
- **Clean Visual Annotations**: Minimized on-screen clutter with color-coded bounding boxes and stable slot IDs.
- **Multi-Source Support**: Handles live video feeds (`carPark.mp4`) and static image analysis (`carpark1.jpg`, `carpark2.jpeg`).

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://https://github.com/MsShivi214/SmartParkAI.git
   cd SmartParkAI
   ```

2. **Install dependencies**:
   ```bash
   pip install flask opencv-python-headless numpy ultralytics easyocr
   ```

3. **Required Assets**:
   Ensure the following files are in the root directory:
   - `carPark.mp4` (Video source)
   - `carpark1.jpg`, `carpark2.jpeg` (Static images)
   - `CarParkPos` (Pre-defined slot coordinates)
   - `yolov8n.pt` (YOLOv8 model weights)

## 🚦 How to Run

### 1. Start the Live Management System
Run the Flask application to launch the web dashboard and analysis engine:
```bash
python app.py
```
Once running, open your browser and navigate to:
**[http://localhost:5000](http://localhost:5000)**

### 2. View the Dashboard
- **Live Video Feed**: Watch the real-time processing of `carPark.mp4` with green/red bounding boxes and slot IDs.
- **Real-Time Table**: Monitor the synchronized list of all slots, their current status, and detected vehicle numbers.
- **Static Analysis**: View processed results for different parking lot layouts.

### 3. Utility Tools (Optional)
If you need to define new parking slots on a custom image:
```bash
python ParkingSpacePicker.py
```
- **Left Click**: Add a space.
- **Right Click**: Remove a space.
- Coordinates are saved to `CarParkPos`.

## 📂 Project Structure

- `app.py`: The main Flask server and AI analysis engine.
- `templates/index.html`: Modern dashboard UI with real-time data synchronization.
- `main.py`: Original CLI-based detection script.
- `ParkingSpacePicker.py`: Tool for manual coordinate mapping.
- `CarParkPos`: Pickle file storing the spatially sorted slot coordinates.
- `static/`: Contains live-updated result snapshots.

## 🧠 How it Works

1. **Spatial Sorting**: During startup, slot coordinates from `CarParkPos` are sorted lane-by-lane to establish a fixed 1-to-N numbering system.
2. **Hybrid Detection**:
   - **YOLOv8**: Confirms occupancy if a vehicle overlaps with a slot by at least 40%.
   - **Pixel Density**: Measures ground changes within the slot ROI to catch vehicles YOLO might miss.
3. **Temporal Filtering**: Statuses are stabilized over a 5-frame window to ensure a flicker-free "live" experience.
4. **OCR Pipeline**: EasyOCR extracts text from occupied slots, with a filtering layer to identify valid license plate formats.

## 🤝 Credits
Developed as an advanced smart-city solution for automated parking management.
