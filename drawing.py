import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1)

# Initialize Pygame
pygame.init()
infoObject = pygame.display.Info()
screen = pygame.display.set_mode(
    (infoObject.current_w, infoObject.current_h), pygame.RESIZABLE
)
pygame.display.set_caption("Finger Drawing")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

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
    running = True
    drawing = False
    prev_point = None
    brush_size = 10
    mouse_pos = (0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

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
            
            
            # If index finger is close to thumb, erase; otherwise, draw
            if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < 25:
                drawing = True
            elif np.linalg.norm(np.array(thumb_pos) - np.array(pinkie_pos)) < 25:
                drawing = False
            else:
                drawing = 'none'


            if prev_point is not None:
                # Interpolate mouse movement
                mouse_pos = (
                    int(0.9 * mouse_pos[0] + 0.1 * index_pos[0]),
                    int(0.9 * mouse_pos[1] + 0.1 * index_pos[1])
                )
            else:
                mouse_pos = index_pos
                prev_point = mouse_pos
            
            if drawing and drawing != 'none':
                pygame.draw.circle(screen, RED, index_pos, brush_size)
                if prev_point:
                    pygame.draw.line(screen, RED, prev_point, index_pos, brush_size * 2)
                prev_point = index_pos
            elif not drawing:
                pygame.draw.circle(screen, BLACK, index_pos, brush_size*4)
                #if prev_point:
                #    pygame.draw.line(screen, BLACK, prev_point, index_pos, brush_size * 4)
                prev_point = index_pos
            pygame.mouse.set_pos(index_pos)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    # Initialize webcam
    cap = cv2.VideoCapture(0)
        # Get display resolution
    display_info = pygame.display.Info()
    display_width = display_info.current_w
    display_height = display_info.current_h
     # Set webcam frame size to match display resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    main()


#import cv2
#import mediapipe as mp
#import numpy as np
#import pygame
#
## Initialize MediaPipe Hands model
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
#
## Initialize Pygame
#pygame.init()
#infoObject = pygame.display.Info()
#screen = pygame.display.set_mode(
#    (infoObject.current_w, infoObject.current_h), pygame.RESIZABLE
#)
#pygame.display.set_caption("Finger Drawing")
#clock = pygame.time.Clock()
#
## Colors
#BLACK = (0, 0, 0)
#WHITE = (255, 255, 255)
#RED = (255, 0, 0)
#
#def get_hand_landmarks(image):
#    # Convert image to RGB
#    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#    # Flip image horizontally
#    image_rgb = cv2.flip(image_rgb, 1)
#
#    # Process image with MediaPipe Hands
#    results = hands.process(image_rgb)
#
#    # Get hand landmarks
#    landmark_list = []
#    if results.multi_hand_landmarks:
#        for hand_landmarks in results.multi_hand_landmarks:
#            landmark_list.append(hand_landmarks)
#
#    return landmark_list
#
#def main():
#    running = True
#    drawing = False
#    prev_point = None
#    brush_size = 10
#    mouse_pos = (0, 0)
#    current_hand = None
#
#    while running:
#        for event in pygame.event.get():
#            if event.type == pygame.QUIT:
#                running = False
#            elif event.type == pygame.MOUSEBUTTONDOWN:
#                drawing = True
#            elif event.type == pygame.MOUSEBUTTONUP:
#                drawing = False
#
#        # Get hand landmarks from webcam
#        _, frame = cap.read()
#        hand_landmarks = get_hand_landmarks(frame)
#
#        # Draw or erase based on hand position
#        if hand_landmarks:
#            leftmost_x = min([min(landmark.x for landmark in hand_landmarks[i].landmark) for i in range(len(hand_landmarks))])
#            rightmost_x = max([max(landmark.x for landmark in hand_landmarks[i].landmark) for i in range(len(hand_landmarks))])
#
#            if leftmost_x < 0.5 and current_hand != "left":
#                current_hand = "left"
#                mouse_pos = (0, 0)  # Reset mouse position when changing hands
#            elif rightmost_x > 0.5 and current_hand != "right":
#                current_hand = "right"
#                mouse_pos = (0, 0)  # Reset mouse position when changing hands
#
#            for hand_landmark in hand_landmarks:
#                index_tip = hand_landmark.landmark[8]
#                index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
#
#                thumb_tip = hand_landmark.landmark[4]
#                thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
#
#                pinkie_tip = hand_landmark.landmark[20]
#                pinkie_pos = (int(pinkie_tip.x * frame.shape[1]), int(pinkie_tip.y * frame.shape[0]))
#
#                # Determine if it's left or right hand
#                if current_hand == "left":
#                    if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < 25:
#                        drawing = False
#                    elif np.linalg.norm(np.array(thumb_pos) - np.array(pinkie_pos)) < 25:
#                        drawing = True
#                    else:
#                        drawing = 'none'
#                elif current_hand == "right":
#                    if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < 25:
#                        drawing = True
#                    elif np.linalg.norm(np.array(thumb_pos) - np.array(pinkie_pos)) < 25:
#                        drawing = False
#                    else:
#                        drawing = 'none'
#
#                if prev_point is not None:
#                    # Interpolate mouse movement
#                    mouse_pos = (
#                        int(0.9 * mouse_pos[0] + 0.1 * index_pos[0]),
#                        int(0.9 * mouse_pos[1] + 0.1 * index_pos[1])
#                    )
#                else:
#                    mouse_pos = index_pos
#                    prev_point = mouse_pos
#
#                if drawing and drawing != 'none':
#                    pygame.draw.circle(screen, RED, index_pos, brush_size)
#                    if prev_point:
#                        pygame.draw.line(screen, RED, prev_point, index_pos, brush_size * 2)
#                    prev_point = index_pos
#                elif not drawing:
#                    pygame.draw.circle(screen, BLACK, index_pos, brush_size*4)
#                    prev_point = index_pos
#                pygame.mouse.set_pos(index_pos)
#
#        pygame.display.flip()
#        clock.tick(30)
#
#    pygame.quit()
#
#if __name__ == "__main__":
#    # Initialize webcam
#    cap = cv2.VideoCapture(0)
#    main()
#