import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to recognize gestures


def recognize_gesture(landmarks):
    thumb_is_open = False
    index_is_open = False
    middle_is_open = False
    ring_is_open = False
    pinky_is_open = False
    #TODO:change parameters for better performance
    # Check if fingers are open
    if landmarks[4].x < landmarks[3].x:  # Thumb
        thumb_is_open = True
    if landmarks[8].y < landmarks[6].y:  # Index finger
        index_is_open = True
    if landmarks[12].y < landmarks[10].y:  # Middle finger
        middle_is_open = True
    if landmarks[16].y < landmarks[14].y:  # Ring finger
        ring_is_open = True
    if landmarks[20].y < landmarks[18].y:  # Pinky finger
        pinky_is_open = True

    # Identify gestures
    if thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Thumbs Up"
    elif index_is_open and middle_is_open and not ring_is_open and not pinky_is_open and thumb_is_open:
        return "Peace Sign"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Fist"
    else:
        return "Unknown Gesture"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Draw the hand landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the gesture
            gesture = recognize_gesture(hand_landmarks.landmark)
            cv2.putText(image, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
