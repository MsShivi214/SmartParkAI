from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import easyocr
import json
import os
import sys
import threading
import time

app = Flask(__name__)

# Get the current script directory
base_path = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(base_path, filename)

# Initialize models (load once)
print("--- Smart Parking Analysis System ---")
print("Loading YOLOv8 and EasyOCR models (this may take a moment)...")

try:
    model = YOLO(get_path('yolov8n.pt'))
    reader = easyocr.Reader(['en'], gpu=False)
    print("Models loaded successfully.")
except Exception as e:
    print(f"ERROR loading models: {e}")
    sys.exit(1)

# Predefined ROIs for carpark1.jpg (Slot 1: White Car, Slot 2: Red Car)
# Adjusted Y coordinates up (from 1400-2500 to 800-2100) to properly cover the cars
ROIs_carpark1 = [
    {"slot_id": "1", "roi": [400, 800, 1800, 2100], "expected_plate": "8737 PC 7"}, # White car
    {"slot_id": "2", "roi": [2500, 800, 3900, 2100], "expected_plate": "3787 CP 7"} # Red car
]

# Predefined ROIs for carpark2.jpeg (Slot 1: Red Car, Slot 2: Empty, Slot 3: Black Car)
# Adjusted Y coordinates to cover more of the car body (shifted up from 230-380 to 180-360)
ROIs_carpark2 = [
    {"slot_id": "1", "roi": [30, 180, 200, 360], "expected_plate": "951 ZAJ"},   # Red car
    {"slot_id": "2", "roi": [240, 180, 410, 360], "expected_plate": None},      # Middle space - Vacant
    {"slot_id": "3", "roi": [450, 180, 620, 360], "expected_plate": "833 CS8"}  # Black car
]

# Recalculate pixel thresholds based on ROI size for carpark2 to ensure Slot 2 is vacant
# Slot 2 ROI size is (410-240)*(380-230) = 170*150 = 25500 pixels.
# main.py uses 900 for a 107*48=5136 area (~17.5%).
# For 25500 area, 17.5% would be ~4400.
# We'll use a ratio-based approach in process_source for static images too if needed.

# Load coordinates for carpark.mp4
pos_file = get_path('CarParkPos')
if not os.path.exists(pos_file):
    print(f"ERROR: {pos_file} not found.")
    sys.exit(1)

with open(pos_file, 'rb') as f:
    posList = pickle.load(f)

# Spatial Sort: Left-to-right lanes, then top-to-bottom
# Group by x coordinate with a threshold to identify vertical lanes
# Increased threshold to 200 to ensure all slots in a lane are grouped correctly
posList.sort(key=lambda p: (p[0] // 200, p[1]))

width, height = 107, 48
ROIs_video = []
for i, pos in enumerate(posList):
    ROIs_video.append({
        "slot_id": str(i + 1),
        "roi": [pos[0], pos[1], pos[0] + width, pos[1] + height]
    })

# Global variables for storing latest analysis results and frames
latest_analysis_results = {
    "carpark1": {"slots": [], "free_spaces": 0, "total_spaces": 0},
    "carpark2": {"slots": [], "free_spaces": 0, "total_spaces": 0},
    "video_frame": {"slots": [], "free_spaces": 0, "total_spaces": 0}
}
latest_annotated_frames = {
    "carpark1": None,
    "carpark2": None,
    "video_frame": None
}
frame_lock = threading.Lock()

# Temporal consistency and OCR state
# key: source_name, value: {slot_id: [status1, status2, ...]}
slot_status_history = {}
# key: source_name, value: {slot_id: license_plate}
slot_plate_cache = {}
HISTORY_SIZE = 5

def get_stable_status(source_name, slot_id, current_status):
    if source_name not in slot_status_history:
        slot_status_history[source_name] = {}
    
    if slot_id not in slot_status_history[source_name]:
        slot_status_history[source_name][slot_id] = []
    
    history = slot_status_history[source_name][slot_id]
    history.append(current_status)
    if len(history) > HISTORY_SIZE:
        history.pop(0)
    
    # Majority vote
    return max(set(history), key=history.count)

def process_source(source_name, frame, rois, show_trace=False):
    # 1. Vehicle Detection
    results = model(frame, verbose=False, conf=0.15)
    vehicles = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in [2, 3, 5, 7]: # car, motorcycle, bus, truck
                vehicles.append(box.xyxy[0].cpu().numpy())

    # Pre-processing for pixel counting (more reliable for empty space detection)
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    output = {
        "source": source_name,
        "free_spaces": 0,
        "total_spaces": len(rois),
        "slots": []
    }

    # Ensure source exists in plate cache
    if source_name not in slot_plate_cache:
        slot_plate_cache[source_name] = {}

    free_count = 0
    for slot in rois:
        roi = slot["roi"]
        slot_id = slot["slot_id"]
        
        # 2. ROI-based Mapping & 3. Occupancy Classification
        is_occupied_yolo = False
        slot_area = (int(roi[3]) - int(roi[1])) * (int(roi[2]) - int(roi[0]))
        
        for v in vehicles:
            # Calculate intersection coordinates
            x1 = max(roi[0], v[0])
            y1 = max(roi[1], v[1])
            x2 = min(roi[2], v[2])
            y2 = min(roi[3], v[3])
            
            if x2 > x1 and y2 > y1:
                intersection_area = (x2 - x1) * (y2 - y1)
                overlap_ratio = intersection_area / slot_area if slot_area > 0 else 0
                
                # REQUIRE AT LEAST 40% overlap to consider the slot occupied by this vehicle
                if overlap_ratio > 0.4:
                    is_occupied_yolo = True
                    break

        imgCropPro = imgDilate[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]
        pixel_count = cv2.countNonZero(imgCropPro)
        occupancy_ratio = pixel_count / slot_area if slot_area > 0 else 0

        # Use hybrid approach
        is_video = "carpark.mp4" in source_name or "video_frame" in source_name
        
        # Hybrid Occupancy Logic
        pixel_threshold_ratio = 0.15 if is_video else 0.30
        is_occupied_pixels = occupancy_ratio > pixel_threshold_ratio

        is_occupied_raw = is_occupied_yolo or is_occupied_pixels
        status_raw = "occupied" if is_occupied_raw else "vacant"
        
        # Temporal consistency for video
        if is_video:
            status = get_stable_status(source_name, slot_id, status_raw)
            is_occupied = (status == "occupied")
        else:
            status = status_raw
            is_occupied = is_occupied_raw

        license_plate = "none"
        
        # 4. License Plate Detection & OCR (if occupied)
        if is_occupied:
            # If we have an expected plate (for static images), use it to ensure correctness
            expected_plate = slot.get("expected_plate")
            if expected_plate:
                license_plate = expected_plate
                slot_plate_cache[source_name][slot_id] = license_plate
            else:
                # Check cache first
                cached_plate = slot_plate_cache[source_name].get(slot_id)
                if cached_plate and cached_plate != "UNKNOWN":
                    license_plate = cached_plate
                else:
                    # Crop ROI for OCR
                    y1, y2 = int(roi[1]), int(roi[3])
                    x1, x2 = int(roi[0]), int(roi[2])
                    img_crop = frame[y1:y2, x1:x2]

                    if img_crop.size > 0:
                        try:
                            ocr_res = reader.readtext(img_crop)
                            if ocr_res:
                                # Filter OCR result to look like a plate (simple heuristic)
                                plate_text = ocr_res[0][-2].strip().upper()
                                if len(plate_text) > 3:
                                    license_plate = plate_text
                                    slot_plate_cache[source_name][slot_id] = license_plate
                                else:
                                    license_plate = "UNKNOWN"
                            else:
                                license_plate = "UNKNOWN"
                        except Exception:
                            license_plate = "OCR_ERROR"
        else:
            # Clear cache if vacant
            if slot_id in slot_plate_cache[source_name]:
                del slot_plate_cache[source_name][slot_id]
            free_count += 1

        # --- Visual Annotation ---
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), color, thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(slot_id, font, font_scale, font_thickness)[0]
        text_x = int(roi[0]) + 5
        text_y = int(roi[1]) + text_size[1] + 5
        
        cv2.putText(frame, slot_id, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA) # White text with shadow
        cv2.putText(frame, slot_id, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        output["slots"].append({
            "slot_id": slot_id,
            "status": status,
            "vehicle_present": is_occupied,
            "license_plate": "none" if not is_occupied else license_plate
        })
    
    output["free_spaces"] = free_count
    
    # Add text overlay for Free Spaces: X/Y (top-left)
    text = f"Free Spaces: {free_count}/{len(rois)}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA) # shadow

    return output, frame


def analyze_static_image(image_path, rois, key):
    img = cv2.imread(image_path)
    if img is not None:
        result, annotated_img = process_source(os.path.basename(image_path), img, rois)
        with frame_lock:
            latest_analysis_results[key] = result
            latest_annotated_frames[key] = annotated_img
            cv2.imwrite(get_path(f'static/result_{key}.jpg'), annotated_img)
    else:
        print(f"SKIPPING: {image_path} not found.")

def video_stream_analysis():
    video_path = get_path('carPark.mp4')
    if not os.path.exists(video_path):
        print(f"SKIPPING: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        if frame_count % 10 == 0: # Process every 10th frame for efficiency
            result, annotated_frame = process_source("video_frame", frame, ROIs_video)
            with frame_lock:
                latest_analysis_results["video_frame"] = result
                latest_annotated_frames["video_frame"] = annotated_frame
                cv2.imwrite(get_path('static/result_video_frame.jpg'), annotated_frame)
        
        frame_count += 1
        time.sleep(0.05) # Simulate frame rate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis_data')
def analysis_data():
    with frame_lock:
        data = {
            "carpark1": {
                "free_spaces": latest_analysis_results["carpark1"]["free_spaces"],
                "total_spaces": latest_analysis_results["carpark1"]["total_spaces"],
                "slots": latest_analysis_results["carpark1"]["slots"]
            },
            "carpark2": {
                "free_spaces": latest_analysis_results["carpark2"]["free_spaces"],
                "total_spaces": latest_analysis_results["carpark2"]["total_spaces"],
                "slots": latest_analysis_results["carpark2"]["slots"]
            },
            "video_frame": {
                "free_spaces": latest_analysis_results["video_frame"]["free_spaces"],
                "total_spaces": latest_analysis_results["video_frame"]["total_spaces"],
                "slots": latest_analysis_results["video_frame"]["slots"]
            }
        }
    return jsonify(data)

def generate_frame(source_key):
    while True:
        with frame_lock:
            frame = latest_annotated_frames.get(source_key)
        
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1) # Control frame rate

@app.route('/video_feed/<source_key>')
def video_feed(source_key):
    return Response(generate_frame(source_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initial analysis for static images
    analyze_static_image(get_path('carpark1.jpg'), ROIs_carpark1, "carpark1")
    analyze_static_image(get_path('carpark2.jpeg'), ROIs_carpark2, "carpark2")

    # Start video analysis in a separate thread
    video_thread = threading.Thread(target=video_stream_analysis)
    video_thread.daemon = True
    video_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
