# Object detection on road
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('helper/yolov3.weights', 'helper/yolov3.cfg')
classes = []
with open('helper/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def process_vehicle_detection(image):
    height, width, _ = image.shape
    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get output from output layers
    detections = net.forward(output_layer_names)

    # Postprocess detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the confidence threshold
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, 'vehicle', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return  image

# Load the image
# original_image = cv2.imread('images2/solidWhiteRight.jpg')
# output_image= process_vehicle_detection(original_image)
# cv2.imshow('Original Vehicle',output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Read Video file and processs
# cap = cv2.VideoCapture('training_data/video.mp4')
# subtracao = cv2.createBackgroundSubtractorMOG2()
# delay = 60
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     frame=process_vehicle_detection(frame)
#     cv2.imshow('Lane Detection',frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()
