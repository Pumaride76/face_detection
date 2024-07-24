# import cv2
# import time
# import mediapipe as mp
# import pygame
# import numpy as np
# from deepface import DeepFace

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize Pygame for audio playback
# pygame.init()

# # Load audio file for happy emotion (example)
# happy_sound = pygame.mixer.Sound('happy.mp3')

# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Define the time interval for emotion update (in seconds)
# update_interval = 2

# # Dictionary to store the last detected emotion and timestamp for each person
# last_emotions = {}

# # Flag to track if happy audio is playing
# happy_playing = False

# # List to store heart animation details
# hearts = []

# # Function to draw a heart shape
# def draw_heart(frame, center, size, color):
#     radius = size // 2
#     offset = radius // 2
#     # Top left circle
#     cv2.circle(frame, (center[0] - offset, center[1] - offset), radius, color, -1)
#     # Top right circle
#     cv2.circle(frame, (center[0] + offset, center[1] - offset), radius, color, -1)
#     # Bottom triangle
#     pts = np.array([[center[0] - size, center[1] - offset], 
#                     [center[0] + size, center[1] - offset], 
#                     [center[0], center[1] + size]], np.int32)
#     cv2.fillPoly(frame, [pts], color)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Resize frame for faster processing
#     frame_height, frame_width, _ = frame.shape
#     resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

#     # Detect emotions in the frame
#     results = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)

#     # Convert the BGR image to RGB for MediaPipe
#     frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

#     # Process the frame for hand gestures
#     hand_results = hands.process(frame_rgb)

#     # Draw bounding boxes and emotion labels on the frame
#     person_count = 0
#     current_time = time.time()  # Get current time once outside the loop
#     for result in results:
#         if person_count >= 4:
#             break

#         region = result['region']
#         dominant_emotion = result['dominant_emotion']

#         bounding_box = (region['x'], region['y'], region['w'], region['h'])

#         # Scale bounding box coordinates to the original frame size
#         bounding_box = (bounding_box[0] * 2, bounding_box[1] * 2, bounding_box[2] * 2, bounding_box[3] * 2)

#         # Check if the person is already in the last_emotions dictionary
#         if person_count in last_emotions:
#             last_emotion, last_time = last_emotions[person_count]
#             if current_time - last_time >= update_interval:
#                 # Update the emotion and timestamp
#                 last_emotions[person_count] = (dominant_emotion, current_time)
#             else:
#                 # Use the last detected emotion
#                 dominant_emotion = last_emotion
#         else:
#             # Add the person to the dictionary with the current emotion and timestamp
#             last_emotions[person_count] = (dominant_emotion, current_time)

#         # Draw bounding box
#         cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
#                       (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#                       (0, 255, 0), 2)

#         # Put the dominant emotion text
#         cv2.putText(frame, dominant_emotion, 
#                     (bounding_box[0], bounding_box[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         # Check if "happy" emotion is detected
#         if dominant_emotion == "happy":
#             # Check if happy audio is not already playing
#             if not happy_playing:
#                 # Play the happy sound
#                 happy_sound.play(-1)  # -1 loops indefinitely
#                 happy_playing = True
#         else:
#             # Stop playing happy sound if another emotion is detected
#             if happy_playing:
#                 happy_sound.stop()
#                 happy_playing = False

#         person_count += 1

#     # Draw hand landmarks and detect gestures
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:

#             thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#             ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#             pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
#             index_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
#             pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

#             # Check for thumbs up gesture
#             if (thumb_tip.y < index_finger_tip.y and
#                 index_finger_tip.y < middle_finger_tip.y and
#                 middle_finger_tip.y < ring_finger_tip.y and
#                 pinky_tip.y > pinky_base.y):
#                 cv2.putText(frame, "Thumbs Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             # Check for heart gesture (index finger tip touching middle finger tip)
#             elif (index_finger_tip.y < middle_finger_tip.y and
#                   index_finger_tip.x > middle_finger_tip.x):
#                 cv2.putText(frame, "Heart", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                 hearts.append({'center': (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])), 'size': 20, 'color': (0, 0, 255), 'start_time': current_time})

#             # Check for waving "Hi" gesture (index finger and middle finger extended, others folded)
#             elif (thumb_tip.y > index_finger_tip.y and
#                   thumb_tip.y > middle_finger_tip.y and
#                   thumb_tip.y > ring_finger_tip.y and
#                   thumb_tip.y > pinky_tip.y):
#                 cv2.putText(frame, "Hi", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Update and draw hearts
#     for heart in hearts:
#         elapsed_time = current_time - heart['start_time']
#         heart['center'] = (heart['center'][0], heart['center'][1] - int(elapsed_time * 100))
#         draw_heart(frame, heart['center'], heart['size'], heart['color'])

#     # Remove hearts that have moved off the screen
#     hearts = [heart for heart in hearts if heart['center'][1] > 0]

#     # Display the resulting frame
#     cv2.imshow('Live Emotion and Gesture Detection', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Stop the happy sound if it's still playing
# if happy_playing:
#     happy_sound.stop()

# # When everything is done, release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import time
import mediapipe as mp
import pygame
import numpy as np
from fer import FER

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio playback
pygame.init()

# Load audio file for happy emotion (example)
happy_sound = pygame.mixer.Sound('happy.mp3')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the time interval for emotion update (in seconds)
update_interval = 2

# Dictionary to store the last detected emotion and timestamp for each person
last_emotions = {}

# Flag to track if happy audio is playing
happy_playing = False

# List to store heart animation details
hearts = []

# Initialize FER detector
emotion_detector = FER(mtcnn=True)

# Function to draw a heart shape
def draw_heart(frame, center, size, color):
    radius = size // 2
    offset = radius // 2
    # Top left circle
    cv2.circle(frame, (center[0] - offset, center[1] - offset), radius, color, -1)
    # Top right circle
    cv2.circle(frame, (center[0] + offset, center[1] - offset), radius, color, -1)
    # Bottom triangle
    pts = np.array([[center[0] - size, center[1] - offset], 
                    [center[0] + size, center[1] - offset], 
                    [center[0], center[1] + size]], np.int32)
    cv2.fillPoly(frame, [pts], color)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for faster processing
    frame_height, frame_width, _ = frame.shape
    resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # Detect emotions in the frame
    emotion_results = emotion_detector.detect_emotions(resized_frame)

    # Convert the BGR image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand gestures
    hand_results = hands.process(frame_rgb)

    # Draw bounding boxes and emotion labels on the frame
    person_count = 0
    current_time = time.time()  # Get current time once outside the loop
    for result in emotion_results:
        if person_count >= 4:
            break

        bounding_box = result['box']
        emotions = result['emotions']
        dominant_emotion = max(emotions, key=emotions.get)

        # Scale bounding box coordinates to the original frame size
        bounding_box = (bounding_box[0] * 2, bounding_box[1] * 2, bounding_box[2] * 2, bounding_box[3] * 2)

        # Check if the person is already in the last_emotions dictionary
        if person_count in last_emotions:
            last_emotion, last_time = last_emotions[person_count]
            if current_time - last_time >= update_interval:
                # Update the emotion and timestamp
                last_emotions[person_count] = (dominant_emotion, current_time)
            else:
                # Use the last detected emotion
                dominant_emotion = last_emotion
        else:
            # Add the person to the dictionary with the current emotion and timestamp
            last_emotions[person_count] = (dominant_emotion, current_time)

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
                hearts.append({'center': (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])), 'size': 20, 'color': (0, 0, 255), 'start_time': current_time})

            # Check for waving "Hi" gesture (index finger and middle finger extended, others folded)
            elif (thumb_tip.y > index_finger_tip.y and
                  thumb_tip.y > middle_finger_tip.y and
                  thumb_tip.y > ring_finger_tip.y and
                  thumb_tip.y > pinky_tip.y):
                cv2.putText(frame, "Hi", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Update and draw hearts
    for heart in hearts:
        elapsed_time = current_time - heart['start_time']
        heart['center'] = (heart['center'][0], heart['center'][1] - int(elapsed_time * 100))
        draw_heart(frame, heart['center'], heart['size'], heart['color'])

    # Remove hearts that have moved off the screen
    hearts = [heart for heart in hearts if heart['center'][1] > 0]

    # Display the resulting frame
    cv2.imshow('Live Emotion and Gesture Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the happy sound if it's still playing
if happy_playing:
    happy_sound.stop()

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
