import tensorflow as tf
import numpy as np
import joblib
import threading
import cv2

# Load models
knn = joblib.load("models/knn_model.pkl")
svm = joblib.load("models/svm_model.pkl")
enc = joblib.load("models/label_model.pkl")

# Feature extractor
base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extractor = tf.keras.models.Model(inputs=base_model.input, outputs=x)
feature_extractor.trainable = False

# Shared variables
current_frame = None
knn_probs = ""
svm_probs = ""
lock = threading.Lock()

BOX_SIZE = 244

def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()

def prediction_thread():
    global current_frame, knn_probs, svm_probs
    while True:
        if current_frame is not None:
            lock.acquire()
            frame_copy = current_frame.copy()
            lock.release()

            h, w, _ = frame_copy.shape
            x1 = w // 2 - BOX_SIZE // 2
            y1 = h // 2 - BOX_SIZE // 2
            x2 = x1 + BOX_SIZE
            y2 = y1 + BOX_SIZE

            cropped = frame_copy[y1:y2, x1:x2]

            features = extract_features_from_frame(cropped)
            
            # Get probabilities
            knn_prob_values = knn.predict_proba([features])[0]  # probabilities for each class
            svm_prob_values = svm.predict_proba([features])[0]

            # Convert to readable string with percentages
            knn_probs = ", ".join([f"{cls}: {prob*100:.1f}%" for cls, prob in zip(enc.classes_, knn_prob_values)])
            svm_probs = ", ".join([f"{cls}: {prob*100:.1f}%" for cls, prob in zip(enc.classes_, svm_prob_values)])

# Start thread
thread = threading.Thread(target=prediction_thread, daemon=True)
thread.start()

# Open webcam
cap = cv2.VideoCapture(1)  # change index if needed
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lock.acquire()
    current_frame = frame.copy()
    lock.release()

    h, w, _ = frame.shape
    x1 = w // 2 - BOX_SIZE // 2
    y1 = h // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE

    # Draw green box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display probabilities
    cv2.putText(frame, f"KNN: {knn_probs}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"SVM: {svm_probs}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
