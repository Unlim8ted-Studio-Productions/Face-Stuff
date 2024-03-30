import cv2

# Function to apply filters on the face
def apply_face_filter(face_img, filter_type, x, y, w, h):
    if filter_type == 'mustache':
        mustache = cv2.imread('mustache.jpg', -1)
        mustache = cv2.resize(mustache, (w, int(h / 2)))
        for i in range(int(h / 2)):
            for j in range(w):
                if mustache[i, j, 2] != 0:  # Alpha channel
                    face_img[y + i + int(h / 2), x + j] = mustache[i, j, 0:2]
    elif filter_type == 'glasses':
        glasses = cv2.imread('glasses.png', -1)
        glasses = cv2.resize(glasses, (w, int(h / 2)))
        for i in range(int(h / 2)):
            for j in range(w):
                if glasses[i, j, 3] != 0:  # Alpha channel
                    face_img[y + i, x + j] = glasses[i, j, 0:3]
    return face_img

def main():
    # Load the pre-trained cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face_img = frame[y:y+h, x:x+w]

            # Display the original face
            cv2.imshow('Original Face', face_img)

            # Wait for user input to select filter
            print("Select a filter:")
            print("1. Mustache")
            print("2. Glasses")
            print("Press 'q' to quit")
            choice = cv2.waitKey(0)

            # Apply the selected filter
            if choice == ord('1'):
                face_img = apply_face_filter(face_img, 'mustache', x, y, w, h)
            elif choice == ord('2'):
                face_img = apply_face_filter(face_img, 'glasses', x, y, w, h)
            elif choice == ord('q'):
                break
            else:
                print("Invalid choice. Please try again.")

            # Display the face with the applied filter
            cv2.imshow('Face with Filter', face_img)

            # Replace the face region in the original frame with the filtered face
            frame[y:y+h, x:x+w] = face_img

        # Display the frame with face filters
        cv2.imshow('Frame with Face Filters', frame)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
