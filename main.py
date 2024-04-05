import tkinter as tk
import cv2
import math
import pyautogui
import mediapipe as mp

class land():
   def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

# Function to check if a hand is in a balled-up position
def is_hand_balled(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    ring_tip = hand_landmarks[16]
    pinky_tip = hand_landmarks[20]

    # Calculate distances between finger tips
    distances = [
        distance(thumb_tip, index_tip),
        distance(index_tip, middle_tip),
        distance(middle_tip, ring_tip),
        distance(ring_tip, pinky_tip)
    ]

    # If all fingers are close to the thumb, hand is balled up
    return all(d < 30 for d in distances)

# Function to calculate angle between three points
def calculate_angle(p1, p2, p3):
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    return math.degrees(radians)

# Function to calculate dot position based on hand landmarks and screen dimensions
def calculate_dot_position(hand_landmarks, screen_width, screen_height):
    # Define landmarks for hand angle calculation
    index_finger_tip = hand_landmarks[8]
    middle_finger_tip = hand_landmarks[12]
    wrist = hand_landmarks[0]

    # Calculate the angle of the hand
    angle = calculate_angle(index_finger_tip, middle_finger_tip, wrist)

    # Convert angle to screen coordinates
    dot_x = int((angle / 180) * screen_width)
    dot_y = int((0.5 * screen_height))  # Y-coordinate remains at the center of the screen

    return dot_x, dot_y


def Landmark2List(landmarks):
    landmarks = landmarks.landmark
    l = []
    for ll in landmarks:
        l.append(land(ll.x,ll.y,ll.z))
    return l
        
def detect_hand():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.attributes("-transparent", "blue")
    root.attributes("-fullscreen", True)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    dot_canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='blue', highlightthickness=0)
    dot_canvas.pack()

    dot = dot_canvas.create_oval(0, 0, 0, 0, outline='red', width=2)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)

    is_dragging = False
    is_right_click = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_hand_balled(hand_landmarks.landmark):
                    # Click when hand is balled up
                    pyautogui.click()

                dot_x, dot_y = calculate_dot_position(hand_landmarks.landmark, screen_width, screen_height)
                pyautogui.moveTo(dot_x, dot_y)

                dot_canvas.coords(dot, dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5)

                if is_dragging:
                    pyautogui.dragTo(dot_x, dot_y)

                if is_right_click:
                    pyautogui.rightClick()

        root.update()

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()

if __name__ == "__main__":
    detect_hand()