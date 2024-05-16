import cv2
import mediapipe as mp
import numpy as np

# Function to update parameters based on trackbar values
def update_parameters(value):
    global feathering, smoothing
    feathering = cv2.getTrackbarPos('Feathering', 'Settings')
    smoothing = cv2.getTrackbarPos('Smoothing', 'Settings')

def main():
    global feathering, smoothing

    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create window for settings
    cv2.namedWindow('Settings')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Feathering', 'Settings', 20, 100, update_parameters)
    cv2.createTrackbar('Smoothing', 'Settings', 82, 100, update_parameters)

    # Set initial parameter values
    feathering = 20
    smoothing = 82

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Segment the frame into foreground and background
        results = selfie_segmentation.process(frame_rgb)

        # Convert segmentation mask to binary mask
        mask = results.segmentation_mask

        # Convert mask to data type uint8
        mask = np.uint8(mask)

        # Apply the binary mask to the frame to get the foreground
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Apply feathering to the mask
        if feathering > 0:
            kernel = cv2.getGaussianKernel(feathering * 2 + 1, feathering)
            mask = cv2.filter2D(mask, -1, kernel)

        # Smooth the mask
        if smoothing > 0:
            mask = cv2.GaussianBlur(mask, (smoothing * 2 + 1, smoothing * 2 + 1), 0)

        # Apply the modified mask to the frame
        foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the resulting frame with only the foreground (person)
        cv2.imshow('Selfie Segmentation', foreground)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
