import cv2
import numpy as np
def annotate_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

    # Iterate over the detected lines and annotate them
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the annotated image
    cv2.imshow('Annotated Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Replace 'your_image_path.jpg' with the path to your image
annotate_lines('image/img5.png');