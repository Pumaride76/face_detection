import cv2
from fer import FER
import matplotlib.pyplot as plt

# Load the image
image_path = 'happy.jpeg'
image = cv2.imread(image_path)

# Initialize the FER detector
detector = FER(mtcnn=True)

# Detect emotions in the image
results = detector.detect_emotions(image)

# Display the results
for result in results:
    print(result)

# Optionally, display the image with detected emotions
for result in results:
    bounding_box = result['box']
    emotions = result['emotions']
    dominant_emotion = max(emotions, key=emotions.get)

    # Draw bounding box
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), 
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                  (0, 255, 0), 2)

    # Put the dominant emotion text
    cv2.putText(image, dominant_emotion, (bounding_box[0], bounding_box[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert BGR image to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
