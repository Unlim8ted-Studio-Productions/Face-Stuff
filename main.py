import tkinter as tk
import dlib
import pyautogui
import cv2
import math

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    return math.degrees(radians)

# Function to calculate dot position based on eye landmarks and screen dimensions
def calculate_dot_position(left_eye_landmarks, right_eye_landmarks, screen_width, screen_height):
    # Define landmarks for eye angle calculation
    left_eye_angle_landmarks = [36, 37, 38, 39, 40, 41]
    right_eye_angle_landmarks = [42, 43, 44, 45, 46, 47]

    # Calculate the angle of each eye
    left_eye_angle = calculate_angle(left_eye_landmarks[0], left_eye_landmarks[3], left_eye_landmarks[5])
    right_eye_angle = calculate_angle(right_eye_landmarks[0], right_eye_landmarks[3], right_eye_landmarks[5])

    # Calculate the average of the angles
    average_angle = (left_eye_angle + right_eye_angle) / 2

    # Convert average angle to screen coordinates
    dot_x = int((average_angle / 180) * screen_width)
    dot_y = int((0.5 * screen_height))  # Y-coordinate remains at the center of the screen

    return dot_x, dot_y

def detect_eyes():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.attributes("-transparent", "blue")
    root.attributes("-fullscreen", True)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    dot_canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='blue', highlightthickness=0)
    dot_canvas.pack()

    dot = dot_canvas.create_oval(0, 0, 0, 0, outline='red', width=2)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_landmarks = [landmarks.part(i) for i in range(36, 42)]
            right_eye_landmarks = [landmarks.part(i) for i in range(42, 48)]

            dot_x, dot_y = calculate_dot_position(left_eye_landmarks, right_eye_landmarks, screen_width, screen_height)
            pyautogui.moveTo(dot_x, dot_y)

            dot_canvas.coords(dot, dot_x-5, dot_y-5, dot_x+5, dot_y+5)

        root.update()

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()

if __name__ == "__main__":
    detect_eyes()
