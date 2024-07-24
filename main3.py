import cv2
from fer import FER
import time
import mediapipe as mp
import pygame

# Initialize the FER detector
detector = FER(mtcnn=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio playback
pygame.init()

# Load audio file for happy emotion (example)
happy_sound = pygame.mixer.Sound('happy.mp3')

# Set the playback speed (0.5 is half speed, 1.0 is normal speed)
# happy_sound.set_speed(0.5)

# Open the prerecorded video file
video_path = 'manohar.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'.")
    exit()

# Define the time interval for emotion update (in seconds)
update_interval = 2

# Dictionary to store the last detected emotion and timestamp for each person
last_emotions = {}

# Flag to track if happy audio is playing
happy_playing = False

while True:
    # Capture frame-by-frame from the video
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Detect emotions in the frame
    results = detector.detect_emotions(resized_frame)

    # Convert the BGR image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand gestures
    hand_results = hands.process(frame_rgb)

    # Draw bounding boxes and emotion labels on the frame
    person_count = 0
    for idx, result in enumerate(results):
        if person_count >= 2:
            break

        bounding_box = result['box']
        emotions = result['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        current_time = time.time()

        # Check if the person is already in the last_emotions dictionary
        if idx in last_emotions:
            last_emotion, last_time = last_emotions[idx]
            if current_time - last_time >= update_interval:
                # Update the emotion and timestamp
                last_emotions[idx] = (dominant_emotion, current_time)
            else:
                # Use the last detected emotion
                dominant_emotion = last_emotion
        else:
            # Add the person to the dictionary with the current emotion and timestamp
            last_emotions[idx] = (dominant_emotion, current_time)

        # Draw bounding box
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 255, 0), 2)

        # Put the dominant emotion text
        cv2.putText(frame, dominant_emotion, 
                    (bounding_box[0], bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check if "happy" emotion is detected
        if dominant_emotion == "happy":
            # Check if happy audio is not already playing
            if not happy_playing:
                # Play the happy sound
                happy_sound.play(-1)  # -1 loops indefinitely
                happy_playing = True
        else:
            # Stop playing happy sound if another emotion is detected
            if happy_playing:
                happy_sound.stop()
                happy_playing = False

        person_count += 1

    # Draw hand landmarks and detect gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            index_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            # Check for thumbs up gesture
            if (thumb_tip.y < index_finger_tip.y and
                index_finger_tip.y < middle_finger_tip.y and
                middle_finger_tip.y < ring_finger_tip.y and
                pinky_tip.y > pinky_base.y):
                cv2.putText(frame, "Thumbs Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for heart gesture (index finger tip touching middle finger tip)
            elif (index_finger_tip.y < middle_finger_tip.y and
                  index_finger_tip.x > middle_finger_tip.x):
                cv2.putText(frame, "Heart", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for waving "Hi" gesture (index finger and middle finger extended, others folded)
            elif (thumb_tip.y > index_finger_tip.y and
                  thumb_tip.y > middle_finger_tip.y and
                  thumb_tip.y > ring_finger_tip.y and
                  thumb_tip.y > pinky_tip.y):
                cv2.putText(frame, "Hi", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Prerecorded Emotion and Gesture Detection', frame)

    # Delay for visualization (adjust as needed)
    cv2.waitKey(30)  # Adjust delay to match the frame rate of your video

# Stop the happy sound if it's still playing
if happy_playing:
    happy_sound.stop()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
