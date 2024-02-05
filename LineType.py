import matplotlib.pyplot as plt
import numpy as np
import cv2
def show_image_list(img_list, cols=2, fig_size=(15, 15), img_labels='name', show_ticks=True):
    img_count = len(img_list)
    rows = int(img_count / cols)
    cmap = None
    plt.figure(figsize=fig_size)
    for i in range(0, img_count):
        try:
            img_name = img_labels[i]
        except IndexError:
            img_name = "IMAGE_MISSING"

        plt.subplot(rows, cols, i+1)
        img = img_list[i]
        if len(img.shape) < 3:
            cmap = "gray"

        if not show_ticks:
            plt.xticks([])
            plt.yticks([])

        plt.imshow(img, cmap=cmap)

    plt.tight_layout()
    plt.show()

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])
    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)
    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)

    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    # show_image_list([img, mask,masked_image], cols=3, fig_size=(15, 15),
    #                 img_labels=['original', 'hsl'])
    return masked_image
def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def classify_lines(lines, threshold_length=80):
    solid_lines = []
    dotted_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the length of the line
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Classify lines based on length
        if line_length > threshold_length:
            solid_lines.append(line)
        else:
            dotted_lines.append(line)

    return solid_lines, dotted_lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    solid_lines, dotted_lines = classify_lines(lines)
    draw_separate_lines(line_img, solid_lines, color=(0, 0, 255), thickness=5,label="Solid")  # Red for solid lines
    draw_separate_lines(line_img, dotted_lines, color=(0, 255, 0), thickness=2,label="Dashed")  # Green for dotted lines

    #draw_lines(line_img, lines)
    return line_img

rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]

def draw_separate_lines(image, lines, color = (0, 255, 0), thickness=2, label='Lane'):
    # Draw the lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image
def process_lane_line_detection(image):
    filterimg = color_filter(image)

    interest = roi(filterimg)
    # show_image_list([image, filterimg,interest], cols=3, fig_size=(15, 15),
    #                 img_labels=['original', 'hsl'])

    canny = cv2.Canny(grayscale(interest), 50, 120)
    myline = hough_lines(canny, 1, np.pi / 180, 10, 20, 5)
    weighted_img = cv2.addWeighted(myline, 1, image, 0.8, 0)
    return weighted_img


# Load the image
# original_image = cv2.imread('images2/solidWhiteCurve.jpg')  # Replace 'your_lane_image.jpg' with the path to your image
# frame= process_lane_line_detection(original_image)
# cv2.imshow('Original Lane',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Read Video file and process
# cap = cv2.VideoCapture('images2/solidWhiteRight.mp4')
# subtracao = cv2.createBackgroundSubtractorMOG2()
# delay = 60
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     lane_detected_image = process_lane_line_detection(frame)
#     cv2.imshow('Lane Detection',lane_detected_image)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()