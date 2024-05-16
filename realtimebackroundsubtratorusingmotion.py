import cv2

# Function to perform background subtraction and create a mask
def remove_background(frame, background_subtractor):
    fg_mask = background_subtractor.apply(frame)
    return cv2.bitwise_and(frame, frame, mask=fg_mask)

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a background subtractor object
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Remove background
        foreground = remove_background(frame, background_subtractor)

        # Display the resulting frame
        cv2.imshow('Video with Background Removed', foreground)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
