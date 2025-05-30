
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

FOREHEAD_IDXS = [10, 338, 297, 332, 284]
LEFT_CHEEK_IDXS = [234, 93, 132]
RIGHT_CHEEK_IDXS = [454, 323, 361]

def extract_roi(frame, landmarks, indices):
    h, w = frame.shape[:2]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    x, y = max(min(x_coords), 0), max(min(y_coords), 0)
    w_rect = min(max(x_coords) - x, frame.shape[1] - x)
    h_rect = min(max(y_coords) - y, frame.shape[0] - y)
    return frame[y:y+h_rect, x:x+w_rect], (x, y, w_rect, h_rect)

def is_valid_region(region, min_size=50, min_variance=5):
    return region.size >= min_size and np.var(region) >= min_variance

def extract_features_and_mask_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        forehead, fb = extract_roi(frame, landmarks, FOREHEAD_IDXS)
        left, lb = extract_roi(frame, landmarks, LEFT_CHEEK_IDXS)
        right, rb = extract_roi(frame, landmarks, RIGHT_CHEEK_IDXS)

        regions = [forehead, left, right]
        features = []
        mask = []

        for region in regions:
            if is_valid_region(region):
                mean_rgb = np.mean(region, axis=(0, 1))
                mean_hsv = np.mean(cv2.cvtColor(region, cv2.COLOR_BGR2HSV), axis=(0, 1))
                mask.append(1)
            else:
                mean_rgb = np.zeros(3)
                mean_hsv = np.zeros(3)
                mask.append(0)
            features.extend(mean_rgb)
            features.extend(mean_hsv)

        return np.array(features), np.array(mask), [fb, lb, rb]
    else:
        return None, None, None
