import tkinter as tk
import cv2
import math
import numpy as np
import pyautogui
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

class land():
   def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Function to calculate distance between two points
def distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def get_hand_landmarks(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flip image horizontally
    image_rgb = cv2.flip(image_rgb, 1)

    # Process image with MediaPipe Hands
    results = hands.process(image_rgb)

    # Get hand landmarks
    landmarks = None
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]

    return landmarks

# Function to check if a hand is in a balled-up position
def is_clicking(landmarks, frame):
    if landmarks:
        index_tip = landmarks.landmark[8]
        index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

        thumb_tip = landmarks.landmark[4]
        thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
        # If index finger is close to thumb, click
        if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < 10:
            return False
        else:
            return True
    return False

def is_right_clicking(landmarks, frame):
    if landmarks:
        thumb_tip = landmarks.landmark[4]
        thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))

        middle_tip = landmarks.landmark[12]
        middle_pos = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
        # If index finger is close to thumb, click
        if np.linalg.norm(np.array(thumb_pos) - np.array(middle_pos)) < 10:
            return False
        else:
            return True
    return False
    
def get_finger_pos(landmarks, frame):
    if landmarks:
        index_tip = landmarks.landmark[8]
        index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
        return index_pos
    return (500,500)
        
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
    reallyoldclick=False
    oldclick = False
    newclick=False
    old_pos = (500,500)
    new_pos = (500,500)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Get hand landmarks from webcam
        _, frame = cap.read()
        landmarks = get_hand_landmarks(frame)
        is_right_click = is_right_clicking(landmarks, frame)
        reallyoldclick=oldclick
        oldclick=newclick
        newclick=is_clicking(landmarks, frame)
        old_pos = new_pos
        #new_pos=get_finger_pos(landmarks, frame)
        #if oldclick and newclick:
        #    pyautogui.dragTo(new_pos[0], new_pos[1])
        #pyautogui.moveTo(new_pos[0], new_pos[1])
        #if newclick:
        #    pyautogui.leftClick()
        #if reallyoldclick and newclick and not oldclick:
        #    pyautogui.doubleClick()
        #if is_right_click:
        #    pyautogui.rightClick()
        

        root.update()

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()

if __name__ == "__main__":
    detect_hand()