# Object detection on road
import cv2

from CarDetection import process_vehicle_detection
from LaneDetection import process_lane_detection

# Load the image
original_image = cv2.imread('images2/solidWhiteRight.jpg')
car_detetced_image = process_vehicle_detection(original_image)
lane_detected_image = process_lane_detection(car_detetced_image)
cv2.imshow('Original2',lane_detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read Video file and processs
# cap = cv2.VideoCapture('training_data/video.mp4')
# subtracao = cv2.createBackgroundSubtractorMOG2()
# delay = 60
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     car_detetced_image = process_vehicle_detection(frame)
#     lane_detected_image = process_lane_detection(car_detetced_image)
#     cv2.imshow('Lane Detection',lane_detected_image)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()
