from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from facenet_pytorch import MTCNN
from collections import deque, Counter, defaultdict
import time
import os

app = Flask(__name__)

# ------------------------------ AYARLAR ------------------------------
MODEL_PATH = "models/3dcnn_emotion_model.h5"
NUM_FRAMES = 10      
IMG_SIZE = 16        
PREDICTION_INTERVAL = 5
PREDICTION_SMOOTHING_WINDOW = 5
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# ------------------------------ MODEL VE MTCNN ------------------------------
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'SimpleDataAugmentation': tf.keras.layers.Layer})
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# ------------------------------ KAMERA ------------------------------
def gen_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_count = 0
    face_buffers = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
    pred_histories = defaultdict(lambda: deque(maxlen=PREDICTION_SMOOTHING_WINDOW))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Gri ton ve yeniden boyutlandırma
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)) / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)  # kanal ekseni ekle
                face_buffers[idx].append(face_resized)

                # Tahmin
                if len(face_buffers[idx]) == NUM_FRAMES and frame_count % PREDICTION_INTERVAL == 0:
                    clip = np.array(face_buffers[idx]).reshape(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1)
                    preds = model.predict(clip, verbose=0)[0]
                    confidence = np.max(preds)
                    if confidence > 0.3:
                        pred_label = emotion_labels[np.argmax(preds)]
                        pred_histories[idx].append((pred_label, confidence))

                # Son tahmini göster
                if pred_histories[idx]:
                    final_emotion, final_conf = Counter([x[0] for x in pred_histories[idx]]).most_common(1)[0]
                    last_conf = [x[1] for x in pred_histories[idx] if x[0] == final_emotion][-1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{final_emotion} ({last_conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ------------------------------ VİDEO ------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_buffers = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
    pred_histories = defaultdict(lambda: deque(maxlen=PREDICTION_SMOOTHING_WINDOW))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Gri ton ve yeniden boyutlandırma
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)) / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_buffers[idx].append(face_resized)

                # Tahmin
                if len(face_buffers[idx]) == NUM_FRAMES and frame_count % PREDICTION_INTERVAL == 0:
                    clip = np.array(face_buffers[idx]).reshape(1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1)
                    preds = model.predict(clip, verbose=0)[0]
                    confidence = np.max(preds)
                    if confidence > 0.3:
                        pred_label = emotion_labels[np.argmax(preds)]
                        pred_histories[idx].append((pred_label, confidence))

                # Son tahmini göster
                if pred_histories[idx]:
                    final_emotion, final_conf = Counter([x[0] for x in pred_histories[idx]]).most_common(1)[0]
                    last_conf = [x[1] for x in pred_histories[idx] if x[0] == final_emotion][-1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{final_emotion} ({last_conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ------------------------------ FLASK ROUTES ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['video']
        path = os.path.join('uploads', file.filename)
        file.save(path)
        return Response(process_video(path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
