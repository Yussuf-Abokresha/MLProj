import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import threading
import cv2

# Load models
knn = joblib.load("models/knn_model.pkl")
svm = joblib.load("models/svm_model.pkl")
enc = joblib.load("models/label_model.pkl")
scaler = joblib.load("models/scaler_model.pkl")

# Build feature extractor
base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extractor = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=x
)

# Freeze CNN
feature_extractor.trainable = False

# Shared variables
current_frame = None
knn_label = ""
svm_label = ""
lock = threading.Lock()

# Box size
BOX_SIZE = 244

def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

def prediction_thread():
    global current_frame, knn_label, svm_label
    while True:
        if current_frame is not None:
            lock.acquire()
            frame_copy = current_frame.copy()
            lock.release()

            h, w, _ = frame_copy.shape
            # Crop the center 244x244 box
            x1 = w // 2 - BOX_SIZE // 2
            y1 = h // 2 - BOX_SIZE // 2
            x2 = x1 + BOX_SIZE
            y2 = y1 + BOX_SIZE

            cropped = frame_copy[y1:y2, x1:x2]

            features = extract_features_from_frame(cropped)
            features_scaled = scaler.transform([features])
            # ----- KNN prediction with rejection -----
            knn_probs = knn.predict_proba(features_scaled)[0]
            knn_threshold = 0.6  # confidence threshold
            if np.max(knn_probs) < knn_threshold:
                knn_label = "Unknown"  # Unknown class ID
            else:
                knn_label = enc.inverse_transform([np.argmax(knn_probs)])[0]

            # ----- SVM prediction with rejection -----
            svm_probs = svm.predict_proba(features_scaled)[0]
            svm_threshold = 0.6
            if np.max(svm_probs) < svm_threshold:
                svm_label = "Unknown"
            else:
                svm_label = enc.inverse_transform([np.argmax(svm_probs)])[0]

# Start prediction thread
thread = threading.Thread(target=prediction_thread, daemon=True)
thread.start()

# Open webcam (change index if needed)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    lock.acquire()
    current_frame = frame.copy()
    lock.release()

    h, w, _ = frame.shape
    x1 = w // 2 - BOX_SIZE // 2
    y1 = h // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # Draw green box in the center
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display predictions
    cv2.putText(frame, f"KNN: {knn_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"SVM: {svm_label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
