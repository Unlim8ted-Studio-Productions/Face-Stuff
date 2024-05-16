import threading
import cv2
import mediapipe as mp
import numpy as np
import pygame
import pydirectinput
import pyaudio
import vosk
import sys

# Initialize Vosk recognizer
#model_path = "voskmodel"
#vosk.SetLogLevel(-1)  # Suppress log output
model = vosk.Model("vosk-model-small-en-us-0.15")
recognizer = vosk.KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()
pygame.init()
pydirectinput.PAUSE= 0
# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1)

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)



# Define variables for smoothing mouse movement
SMOOTHING_FACTOR = 1  # Adjust this value for different levels of smoothing
NUM_INTERPOLATION_POINTS = 2  # Number of points to interpolate between consecutive frames
prev_index_pos = (0, 0)

def does_dict_have_key(dict, key):
    try:
        dict[key]
        return True
    except KeyError:
        return False
def speech_recognition_thread():
    while True:
        # Get speech input
        data = stream.read(4096)
        
        if len(data) == 0:
            break

        # Recognize speech
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
           # print(result)
            # Type the recognized speech onto the screen
            if "partial" in result:
                recognized_text = result[14:-3]
                #print(recognized_text)
                # Type the recognized speech onto the screen
                pydirectinput.typewrite(recognized_text)
            if "text" in result:
                for i in range(100):
                    pydirectinput.press('delete')
                recognized_text = result[14:-3]
               # print(recognized_text)
                # Type the recognized speech onto the screen
                pydirectinput.typewrite(recognized_text)
                pydirectinput.press('enter')  # Press Enter key after typing
        else:
            print(recognizer.PartialResult())
        
def smooth_mouse_movement(current_pos):
    global prev_index_pos
    # Calculate the difference between current and previous positions
    diff_x = current_pos[0] - prev_index_pos[0]
    diff_y = current_pos[1] - prev_index_pos[1]
    # Smooth the difference based on the smoothing factor
    smoothed_diff_x = int(diff_x * SMOOTHING_FACTOR)
    smoothed_diff_y = int(diff_y * SMOOTHING_FACTOR)
    # Update the previous position
    prev_index_pos = (
        prev_index_pos[0] + smoothed_diff_x,
        prev_index_pos[1] + smoothed_diff_y
    )
    return prev_index_pos


def interpolate_points(start_point, end_point, num_points):
    x_diff = end_point[0] - start_point[0]
    y_diff = end_point[1] - start_point[1]
    interpolated_points = []
    for i in range(1, num_points + 1):
        progress = i / (num_points + 1)
        x = int(start_point[0] + x_diff * progress)
        y = int(start_point[1] + y_diff * progress)
        interpolated_points.append((x, y))
    return interpolated_points

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

def main():
    global prev_index_pos
    running = True
    
    # Get display resolution
    display_info = pygame.display.Info()
    display_width = display_info.current_w
    display_height = display_info.current_h
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Set webcam frame size to match display resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
    leftclick=False
    rightclick=False
    scroll  = False
    # Start speech recognition thread
    speech_thread = threading.Thread(target=speech_recognition_thread)
    speech_thread.daemon = True  # Set as daemon so it exits when the main thread exits
    speech_thread.start()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        leftclick, rightclick, scroll = False, False, False
        
        
        # Get hand landmarks from webcam
        _, frame = cap.read()
        landmarks = get_hand_landmarks(frame)

        # Draw or erase based on hand position
        if landmarks:
            index_tip = landmarks.landmark[8]
            index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

            thumb_tip = landmarks.landmark[4]
            thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))

            pinkie_tip = landmarks.landmark[20]
            pinkie_pos = (int(pinkie_tip.x * frame.shape[1]), int(pinkie_tip.y * frame.shape[0]))
            
            middle_tip = landmarks.landmark[12]
            middle_pos = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
            # If index finger is close to thumb, erase; otherwise, draw
            if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < 25:
                leftclick = True
            elif np.linalg.norm(np.array(thumb_pos) - np.array(pinkie_pos)) < 25:
                rightclick = True
            if np.linalg.norm(np.array(middle_pos) - np.array(thumb_pos)) < 25:
                scroll = True

            # Interpolate mouse movement
            #interpolated_points = interpolate_points(prev_index_pos, index_pos, NUM_INTERPOLATION_POINTS)
            #for pos in interpolated_points:
            p=prev_index_pos
            smoothed_pos = smooth_mouse_movement(index_pos)
                # Move mouse cursor
            pydirectinput.moveTo(smoothed_pos[0], smoothed_pos[1])
            if leftclick:
                pydirectinput.leftClick()
            if rightclick:
                pydirectinput.rightClick()
            if scroll:
                if smoothed_pos[1] < p[1]:
                    pydirectinput.press('down', presses=abs((smoothed_pos[1]-p[1])//2))
                else:
                    pydirectinput.press('up', presses=(smoothed_pos[1]-p[1])//2)
                

if __name__ == "__main__":
    main()
