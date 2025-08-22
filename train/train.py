import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# ------------------------- Ayarlar -------------------------
SEQ_LEN = 10
IMG_SIZE = 16
DATASET_DIR = "EmotionDetection/extracted_faces"

# ------------------------- Veri Hazırlama -------------------------
emotion_labels = {}
X_data, y_data = [], []
label_idx = 0

def load_video_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        frames.append(resized)
    cap.release()

    sequences = []
    for i in range(len(frames) - SEQ_LEN + 1):
        seq = np.array(frames[i:i+SEQ_LEN])
        sequences.append(seq.reshape(SEQ_LEN, IMG_SIZE, IMG_SIZE, 1))
    return sequences

print("Veri yükleniyor...")

for subject in os.listdir(DATASET_DIR):
    subject_path = os.path.join(DATASET_DIR, subject)
    if not os.path.isdir(subject_path): continue

    for emotion in os.listdir(subject_path):
        emotion_path = os.path.join(subject_path, emotion)
        if not os.path.isdir(emotion_path): continue

        if emotion not in emotion_labels:
            emotion_labels[emotion] = label_idx
            label_idx += 1

        for sentence in os.listdir(emotion_path):
            sentence_path = os.path.join(emotion_path, sentence)
            if not os.path.isdir(sentence_path): continue

            for video in os.listdir(sentence_path):
                if not video.lower().endswith(('.mp4', '.avi')): continue
                video_path = os.path.join(sentence_path, video)
                print(f"İşleniyor: {video_path}")
                sequences = load_video_sequence(video_path)
                for seq in sequences:
                    X_data.append(seq)
                    y_data.append(emotion_labels[emotion])

if len(X_data) == 0:
    raise RuntimeError("Hiçbir veri yüklenemedi. Lütfen extracted_faces klasörünü ve içeriğini kontrol edin.")

X_data = np.array(X_data, dtype=np.float32)

# ------------------------- Z-score Normalizasyon -------------------------
mean = np.mean(X_data)
std = np.std(X_data)
X_data = (X_data - mean) / std

y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(emotion_labels))

# ------------------------- Veri Dağılımı Kontrolü -------------------------
label_counts = Counter(np.argmax(y_data, axis=1))
print("Etiket dağılımı:", label_counts)

# ------------------------- Eğitim/Test Bölme -------------------------
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ------------------------- Class Weight Hesaplama -------------------------
class_weights = compute_class_weight(class_weight='balanced',
                                      classes=np.unique(np.argmax(y_data, axis=1)),
                                      y=np.argmax(y_data, axis=1))
class_weight = dict(enumerate(class_weights))

# ------------------------- Model Tanımı -------------------------
def build_3dcnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling3D((1,2,2)),

        tf.keras.layers.Conv3D(64, (3,3,3), activation='relu'),
        tf.keras.layers.MaxPooling3D((1,2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_3dcnn(input_shape=X_train.shape[1:], num_classes=len(emotion_labels))
model.summary()

# ------------------------- Callbacks -------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
]

# ------------------------- Model Eğitimi -------------------------
EPOCHS = 30
BATCH_SIZE = 4

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks
)

# ------------------------- Doğrulukları Bastır -------------------------
print("\n--- Doğruluklar ---")
print("Eğitim doğrulukları:", history.history['accuracy'])
print("Doğrulama doğrulukları:", history.history['val_accuracy'])
print(f"Son epoch eğitim doğruluğu: {history.history['accuracy'][-1]:.4f}")
print(f"Son epoch doğrulama doğruluğu: {history.history['val_accuracy'][-1]:.4f}")

# ------------------------- Grafik Gösterimi -------------------------
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.title("Eğitim vs Doğrulama Doğruluğu")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------- Modeli ve Verileri Kaydet -------------------------
model.save("3dcnn_emotion_model.h5")
np.save("emotion_labels.npy", emotion_labels)
print("Model ve etiketler kaydedildi.")

# ------------------------- Eğitim Geçmişini ve Grafik Verilerini Kaydet -------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history_img32_epoch100.csv", index=False)

np.save("accuracy_values.npy", history.history['accuracy'])
np.save("val_accuracy_values.npy", history.history['val_accuracy'])

print("Eğitim geçmişi ve doğruluk verileri kaydedildi.")
