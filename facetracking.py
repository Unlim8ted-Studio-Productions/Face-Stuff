import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

# Variables to store facial landmark points
landmarks = None
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('Face Tracking', frame)


# Function to track face and detect facial landmarks in real-time
def track_face():
    global landmarks
    global cap
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        
        # Display the frame
        resize = cv2.resize(frame, (cv2.getWindowImageRect('Face Tracking')[2], cv2.getWindowImageRect('Face Tracking')[3])) 
        cv2.imshow('Face Tracking', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


track_face()