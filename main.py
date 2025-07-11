import cv2
from ultralytics import YOLO
import google.generativeai as genai
import os
import textwrap

# === CONFIG: Set your Gemini API key here ===
API_KEY = # insert your API key for gemini
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# === Load YOLOv8 model ===
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt, yolov8m.pt, etc.

# === Gemini image query function ===
def query_gemini(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    prompt = """
This image contains object detections. Please tell the object name that is present in the image and whether there is any violent action going on such as punching, hitting, slapping, etc.
"""
    response = gemini_model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img_data}
    ])
    return response.text

# === List of keywords to detect violent activity ===
VIOLENT_KEYWORDS = [
    "punch", "fight", "slap", "hit", "kick", "attack",
    "violence", "abuse", "strangle", "stab", "shoot"
]

# === Webcam loop ===
cap = cv2.VideoCapture(0)
frame_idx = 0
gemini_display_text = ""
anomaly_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Run YOLO object detection
    results = model(frame)[0]
    object_names = results.names
    boxes = results.boxes

    unknown_detected = False

    # Draw all detected objects with confidence > 0.6
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = object_names[cls_id]

        if conf > 0.6:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            unknown_detected = True

    # If no high-confidence objects or suspicious, query Gemini
    if len(boxes) == 0 or unknown_detected:
        img_path = f"unknown_{frame_idx}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"[⚠] Unknown or suspicious frame {frame_idx}, sending to Gemini...")
        gemini_display_text = query_gemini(img_path)[:500]  # limit response

        # Detect violent activity from Gemini's response
        lower_text = gemini_display_text.lower()
        if any(keyword in lower_text for keyword in VIOLENT_KEYWORDS):
            anomaly_detected = True
            print("[🚨] Anomaly detected: Violent action mentioned in Gemini response.")
        else:
            anomaly_detected = False

    # Display Gemini response
    cv2.putText(frame, "Gemini response (if any) below:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if gemini_display_text:
        wrapped = textwrap.wrap(gemini_display_text, width=60)
        for i, line in enumerate(wrapped[:8]):
            y = 60 + i * 25
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 255), 1)

    # Show anomaly alert
    if anomaly_detected:
        cv2.putText(frame, "ANOMALY DETECTED: VIOLENT ACTIVITY", (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Show the output frame
    cv2.imshow("All Object Detection + Gemini Insight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()