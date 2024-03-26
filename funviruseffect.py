import cv2
import dlib
import pyautogui
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to calculate dot position based on eye landmarks
def calculate_dot_position(eye_landmarks):
    left_eye_midpoint = midpoint(eye_landmarks[0], eye_landmarks[3])
    right_eye_midpoint = midpoint(eye_landmarks[1], eye_landmarks[2])
    dot_x = int((left_eye_midpoint[0] + right_eye_midpoint[0]) / 2)
    dot_y = int((left_eye_midpoint[1] + right_eye_midpoint[1]) / 2)
    return dot_x, dot_y

# Main function to detect eyes and display dot
def main():
    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_TOPMOST, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_TOPMOST, cv2.WINDOW_FULLSCREEN)
        # Capture screen
        screen = pyautogui.screenshot()
        frame = np.array(screen)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_landmarks = [landmarks.part(i) for i in range(36, 42)]
            right_eye_landmarks = [landmarks.part(i) for i in range(42, 48)]

            dot_x, dot_y = calculate_dot_position(left_eye_landmarks + right_eye_landmarks)
            pyautogui.moveTo(dot_x, dot_y)

        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
