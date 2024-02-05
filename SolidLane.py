# Object detection on road
import cv2
from CarDetection import process_vehicle_detection
from LineType import process_lane_line_detection

# Load the image
original_image = cv2.imread('images2/solidWhiteRight.jpg')
car_detected_image = process_vehicle_detection(original_image)
road_highlighted = process_lane_line_detection(car_detected_image)
cv2.imshow('Original',road_highlighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Read Video file and process
# cap = cv2.VideoCapture('images2/solidWhiteRight.mp4')
# subtracao = cv2.createBackgroundSubtractorMOG2()
# delay = 60
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     car_detetced_image = process_vehicle_detection(frame)
#     lane_detected_image = process_lane_line_detection(car_detetced_image)
#     cv2.imshow('Lane Detection',lane_detected_image)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()
