import cv2
import numpy as np
import tensorflow as tf
from facenet_pytorch import MTCNN
import torch
import os
from collections import deque, Counter

# ------------------ Ortam Ayarları ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("PyTorch CUDA kullanılabilir mi:", torch.cuda.is_available())
print("Aktif PyTorch cihazı:", device)

# ------------------ Sabitler ------------------
SEQ_LEN = 10
IMG_SIZE = 16
VIDEO_PATH = "test_video/happiness.avi"
MODEL_PATH = "models/3dcnn_emotion_model.h5"
LABELS_PATH = "labels/emotion_labels.npy"
PREDICTION_SMOOTHING_WINDOW = 1  # Daha anlık sonuç için 1

# ------------------ Model ve Etiketleri Yükle ------------------
model = tf.keras.models.load_model(MODEL_PATH)
emotion_labels = np.load(LABELS_PATH, allow_pickle=True).item()
inv_emotion_labels = {v: k for k, v in emotion_labels.items()}

# ------------------ MTCNN Yüz Algılayıcı ------------------
mtcnn = MTCNN(keep_all=False, device=device)

# ------------------ Video Okuma ------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_buffer = []
pred_history = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])  # İlk yüzü al
        face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        frame_buffer.append(face)

        if len(frame_buffer) >= SEQ_LEN:
            sequence = np.array(frame_buffer[-SEQ_LEN:]).reshape(1, SEQ_LEN, IMG_SIZE, IMG_SIZE, 1) / 255.0
            result = model.predict(sequence, verbose=0)[0]
            label = inv_emotion_labels[np.argmax(result)]
            pred_history.append(label)

            # Anlık tahmin gösterimi
            most_common = Counter(pred_history).most_common(1)[0][0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, most_common, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüle
    cv2.imshow("Duygu Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
