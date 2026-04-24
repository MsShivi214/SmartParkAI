import cv2
import pickle
import cvzone
import numpy as np
from ultralytics import YOLO
import easyocr

# Initialize YOLO model (nano version for speed)
model = YOLO('yolov8n.pt')
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False) # GPU False for compatibility

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

# Dictionary to store plate numbers to avoid redundant OCR
plate_cache = {}

def checkParkingSpace(img, imgPro, yolo_results):
    spaceCounter = 0
    occupied_info = []

    # Extract vehicle boxes from YOLO
    vehicles = []
    if yolo_results:
        for r in yolo_results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]:
                    vehicles.append(box.xyxy[0].cpu().numpy())

    for i, pos in enumerate(posList):
        x, y = pos
        
        # 1. Robust Occupancy Check using Pixel Counting (Original Method)
        imgCropPro = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCropPro)
        
        # Original threshold was 900
        is_occupied_pixels = count >= 900
        
        # 2. Vehicle Confirmation via YOLO (Overlap check)
        roi_box = [x, y, x + width, y + height]
        is_occupied_yolo = False
        for v in vehicles:
            # Check if YOLO box and ROI box overlap
            if not (v[2] < roi_box[0] or v[0] > roi_box[2] or v[3] < roi_box[1] or v[1] > roi_box[3]):
                is_occupied_yolo = True
                break
        
        # Final Occupancy Decision
        is_occupied = is_occupied_pixels

        if not is_occupied:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter += 1
            display_text = f"S{i+1}: Vacant"
        else:
            color = (0, 0, 255)
            thickness = 4
            
            # OCR for License Plate
            if i not in plate_cache:
                if is_occupied_yolo:
                    imgCrop = img[y:y + height, x:x + width]
                    ocr_res = reader.readtext(imgCrop)
                    if ocr_res:
                        plate_cache[i] = ocr_res[0][-2].upper()
                    else:
                        plate_cache[i] = "UNKNOWN"
                else:
                    # If pixels say occupied but YOLO doesn't see a vehicle yet
                    plate_cache[i] = "CAR DETECTED"
            
            plate = plate_cache.get(i, "CAR DETECTED")
            display_text = f"S{i+1}: {plate}"
            occupied_info.append(f"Slot {i+1}: {plate}")

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, display_text, (x, y + height - 3), scale=0.8,
                           thickness=1, offset=2, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))
    
    return spaceCounter, occupied_info

frame_count = 0
results = None

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    if not success: break

    # Pre-processing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Run YOLO detection every 10 frames
    if frame_count % 10 == 0:
        results = model(img, verbose=False, conf=0.15) # Slightly lower conf
    
    free_count, occupied_list = checkParkingSpace(img, imgDilate, results)
    
    # Print and Save Output
    if frame_count % 30 == 0:
        print("\n" + "="*30)
        print("   CAR PARKING STATUS REPORT")
        print("="*30)
        print(f"FREE SPACES: {free_count}/{len(posList)}")
        print(f"OCCUPIED: {len(occupied_list)}")
        print("-" * 30)
        print("LIST OF OCCUPIED SLOTS:")
        for item in occupied_list:
            print(f"  [X] {item}")
        print("="*30 + "\n")
        
        # Save frame to disk for visual verification in headless mode
        cv2.imwrite("parking_status.jpg", img)

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
    frame_count += 1