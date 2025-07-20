import cv2
import mediapipe as mp
import os

# Configuration
SAVE_DIR = r"D:\Graduation Project 2025\Dataset\Continue_dataset"  
CLASS_LABEL = 'Close_look'  # change per session: left_look, forward_look, close_look
PERSON_ID = 'p21'
IMAGE_COUNT = 50
IMAGE_SIZE = 224  # output size       

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks (right eye)
RIGHT_EYE_LANDMARKS = [33, 133]

# Output directory
output_path = os.path.join(SAVE_DIR, CLASS_LABEL, PERSON_ID)
os.makedirs(output_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
captured = 0

print("📷 Press SPACE to capture, Q to quit...")

while cap.isOpened() and captured < IMAGE_COUNT:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get right eye bounding box
            x_coords = [int(face_landmarks.landmark[i].x * w) for i in RIGHT_EYE_LANDMARKS]
            y_coords = [int(face_landmarks.landmark[i].y * h) for i in RIGHT_EYE_LANDMARKS]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Expand box around eye
            margin = 30
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(w, x_max + margin)
            y2 = min(h, y_max + margin)

            # Crop and resize
            eye_crop = frame[y1:y2, x1:x2]
            eye_crop = cv2.resize(eye_crop, (IMAGE_SIZE, IMAGE_SIZE))

            # Show preview
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Eye Preview', eye_crop)
            cv2.imshow('Live Feed', preview)

            key = cv2.waitKey(1)
            if key == ord(' '):  # Space to save
                img_name = f"{CLASS_LABEL}_{PERSON_ID}_{captured:03d}.jpg"
                save_path = os.path.join(output_path, img_name)
                cv2.imwrite(save_path, eye_crop)
                print(f"✅ Saved: {save_path}")
                captured += 1

            elif key == ord('q'):
                break

cap.release() 
cv2.destroyAllWindows()
