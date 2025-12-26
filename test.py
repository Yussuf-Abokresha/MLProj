import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import threading
import cv2
import pandas as pd

# Load models
print("Loading models...")
try:
    svm = joblib.load("models/svm_model.pkl")
    enc = joblib.load("models/label_model.pkl")
    scaler = joblib.load("models/scaler_model.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    

print("Loading ResNet50 feature extractor...")
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


feature_extractor.trainable = False

current_frame = None
svm_label = ""
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
    global current_frame, svm_label
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

            try:
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = frame_copy[y1:y2, x1:x2]

                features = extract_features_from_frame(cropped)
                features_scaled = scaler.transform([features])

               
                svm_probs = svm.predict_proba(features_scaled)[0]
                svm_threshold = 0.6
                if np.max(svm_probs) < svm_threshold:
                    svm_label = "Unknown"
                else:
                    svm_label = enc.inverse_transform([np.argmax(svm_probs)])[0]
            except Exception as e:
                print(f"Prediction thread error: {e}")

def run_gui():
    global current_frame
    
    thread = threading.Thread(target=prediction_thread, daemon=True)
    thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting GUI. Press 'q' to exit.")
    
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

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        cv2.putText(frame, f"SVM: {svm_label}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_batch_prediction(folder_path, output_excel='predictions.xlsx'):
    print(f"Scanning folder: {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    predictions = []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    except Exception as e:
         print(f"Error accessing folder: {e}")
         return

    if not image_files:
        print("No image files found in the folder.")
        return

    print(f"Found {len(image_files)} images. Starting prediction...")

    for img_file in image_files:
        img_full_path = os.path.join(folder_path, img_file)
        
        try:
            img = tf.keras.preprocessing.image.load_img(img_full_path, target_size=(224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.resnet50.preprocess_input(x)
            
            features = feature_extractor.predict(x, verbose=0)
            features = features.flatten().reshape(1, -1)
            
            features_scaled = scaler.transform(features)
            
            pred_idx = svm.predict(features_scaled)[0]
            pred_label = enc.inverse_transform([pred_idx])[0]
            
            predictions.append({'ImageName': img_file, 'predictedlabel': pred_label})
            print(f"Processed: {img_file} -> {pred_label}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            predictions.append({'ImageName': img_file, 'predictedlabel': 'Error'})

    df = pd.DataFrame(predictions)
    try:
        df.to_excel(output_excel, index=False)
        print(f"Predictions saved to {output_excel}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        csv_output = output_excel.replace('.xlsx', '.csv')
        df.to_csv(csv_output, index=False)
        print(f"Saved to CSV instead: {csv_output}")

if __name__ == "__main__":
    print("Select Mode:")
    print("1. CLI GUI (Webcam)")
    print("2. Batch Prediction (Folder -> Excel)")
    print("-" * 30)
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == '1':
        run_gui()
    elif choice == '2':
        folder = input("Enter folder path (default: 'sample'): ").strip()
        if not folder:
            folder = 'sample'
            
        output = input("Enter output Excel path (default: 'output.xlsx'): ").strip()
        if not output:
             output = 'output.xlsx'

        run_batch_prediction(folder, output)
    else:
        print("Invalid choice. Exiting.")
