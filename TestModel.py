import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mask_detection_model.h5", compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("Model loaded successfully!")

# Initialize MediaPipe Face Detection google Ai
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to prepare the face image for prediction
def prepare_image(face, img_size=(224, 224)):
    """ Prepares the detected face image for model prediction. """
    if face is not None and face.size != 0:  # Check if face is not empty
        face = cv2.resize(face, img_size)  # Resize to the model input size
        face = face / 255.0  # Normalize pixel values to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face
    return None  # Return None if no valid face is passed

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # Increased confidence

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        # Convert the frame back to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                # Draw the bounding box around the face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame_bgr.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extract face region for mask prediction
                face = frame_bgr[y:y+h, x:x+w]
                
                # Prepare face for prediction
                prepared_face = prepare_image(face)
                
                if prepared_face is not None:  # Check if face was successfully prepared
                    # Predict mask or no mask
                    prediction = model.predict(prepared_face)[0][0]
                    
                    if prediction > 0.5:
                        label = "No Mask"
                        color = (0, 0, 255)  # Red
                    else:
                        label = "Mask"
                        color = (0, 255, 0)  # Green
                    print(f"Prediction value: {prediction}")
                    # Draw rectangle and label
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame_bgr, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show the frame
        cv2.imshow("Mask Detection", frame_bgr)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty("Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
