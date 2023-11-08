import cv2

# Load the pre-trained object detection model in ONNX format
model = cv2.dnn.readNetFromONNX('D:/School Work/maxx_ai/maxx_ai/tf/best.onnx')

# Read the video frames
cap = cv2.VideoCapture('C:/Users/Asus/OneDrive/Pictures/WIN_20230207_21_36_08_Pro.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (e.g., resize, color conversion)
    resized_frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=True, crop=False)

    # Perform object detection
    model.setInput(blob)
    detections = model.forward()

    # Process the detections (e.g., draw bounding boxes)

    # Display the frame with detections
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()