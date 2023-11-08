import cv2
import numpy as np
import onnxruntime
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load the ONNX model
model_path = "D:/School Work/maxx_ai/maxx_ai/tf/best.onnx"
session = onnxruntime.InferenceSession(model_path)

# Define the threshold for NSFW detection
threshold = 0.5

# Function to preprocess an image
# def preprocess_image(img):
#     img = cv2.resize(img, (224, 224))
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     return img

# def preprocess_image(img):
#     img = cv2.resize(img, (320, 320))
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     return img

def preprocess_image(img):
    img = cv2.resize(img, (320, 320))
    img = img.transpose((2, 0, 1))  # Transpose dimensions to (3, 320, 320)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to classify an image using the ONNX model
def classify_image(img):
    img = preprocess_image(img)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img})
    return predictions[0]

# Function to process frames and classify NSFW content
def process_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Classify the frame
        predictions = classify_image(frame)
        nsfw_score = predictions[0][1]

        # Determine if the frame is NSFW based on the threshold
        is_nsfw = nsfw_score > threshold

        # Do something with the NSFW result
        if is_nsfw:
            print("NSFW content detected!")
            # Perform actions like flagging or filtering the video

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Specify the path to the video file
video_path = "C:/Users/Asus/OneDrive/Pictures/WIN_20230207_21_36_08_Pro.mp4"

# Process the frames and classify NSFW content
process_frames(video_path)