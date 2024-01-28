import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy import stats

# Start of Function
# Convenience function used to show a list of images
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

# Image should have already been converted to HSL color space
def isolate_yellow_hsl(img):
    # Caution - OpenCV encodes the data in ****HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

    return yellow_mask
# Image should have already been converted to HSL color space
def isolate_white_hsl(img):
    # Caution - OpenCV encodes the data in ***HLS*** format
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

    return yellow_mask
def combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white):
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)
def filter_img_hsl(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    return combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)
def get_vertices_for_img(img):
    imshape = img.shape
    img_shape=img.shape
    height = imshape[0]
    width = imshape[1]

    vert = None

    if (width, height) == (960, 540):
        region_bottom_left = (130 ,img_shape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (img_shape[1] - 30,img_shape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200 , 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert

def region_of_interest_chatgpt(img):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # defining vertices for region of interest
    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 2, imshape[0] // 2), (imshape[1], imshape[0])]], dtype=np.int32)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    vert = get_vertices_for_img(img)

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vert, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True,label='Lane'):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img

    # Check if lines is None
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_copy
def separate_lines(lines, img):
    img_shape = img.shape

    middle_x = img_shape[1] / 2

    left_lane_lines = []
    right_lane_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                dx = x2 - x1
                if dx == 0:
                    # Discarding line since we can't gradient is undefined at this dx
                    continue
                dy = y2 - y1

                # Similarly, if the y value remains constant as x increases, discard line
                if dy == 0:
                    continue

                slope = dy / dx

                # This is pure guess than anything...
                # but get rid of lines with a small slope as they are likely to be horizontal one
                epsilon = 0.1
                if abs(slope) <= epsilon:
                    continue

                if slope < 0 and x1 < middle_x and x2 < middle_x:
                    # Lane should also be within the left hand side of region of interest
                    left_lane_lines.append([[x1, y1, x2, y2]])
                elif x1 >= middle_x and x2 >= middle_x:
                    # Lane should also be within the right hand side of region of interest
                    right_lane_lines.append([[x1, y1, x2, y2]])

    return left_lane_lines, right_lane_lines
def color_lanes(img, left_lane_lines, right_lane_lines, left_lane_color=[255, 0, 0], right_lane_color=[0, 0, 255]):
    left_colored_img = draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=True,label="Left")
    right_colored_img = draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False,label="Right")

    return right_colored_img
def find_lane_lines_formula(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

    # Remember, a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)

"""We now define a function to trace a line on the lane:"""

def trace_lane_line(img, lines, top_y, make_copy=True):
    # Check if lines is None or empty
    if lines is None or not lines:
        # No lines to trace, return the original image
        return np.copy(img) if make_copy else img

    A, b = find_lane_lines_formula(lines)
    vert = get_vertices_for_img(img)

    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A

    top_x_to_y = (top_y - b) / A

    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)

def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]

    full_left_lane_img = trace_lane_line(img, left_lane_lines, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line(full_left_lane_img, right_lane_lines, region_top_left[1], make_copy=False)

    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    img_with_lane_weight =  cv2.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)

    return img_with_lane_weight
def canny_edge_detector(blurred_img, low_threshold, high_threshold):
    return cv2.Canny(blurred_img, low_threshold, high_threshold)
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def gaussian_blur(grayscale_img, kernel_size=3):
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)
# End of Function

def process_lane_detection(original_image):
    img_shape = original_image.shape
    hsl_images = cv2.cvtColor(original_image, cv2.COLOR_RGB2HLS)

    hsl_yellow_images = isolate_yellow_hsl(hsl_images)
    hsl_white_images = isolate_white_hsl(hsl_images)
    combined_images = filter_img_hsl(original_image)

    gray_images = grayscale(combined_images)

    kernel_size = 5  # 5, 21
    blurred_images = gaussian_blur(gray_images, kernel_size)
    low_threshold = 50  # 0,10
    high_threshold = 150  # 10,50
    canny_images = canny_edge_detector(blurred_images, low_threshold, high_threshold)

    segmented_images = region_of_interest_chatgpt(canny_images)

    rho = 1
    # 1 degree
    theta = (np.pi / 180) * 1
    threshold = 15
    min_line_length = 20
    max_line_gap = 10
    #hough_lines = hough_transform(segmented_images, rho, theta, threshold, min_line_length, max_line_gap)

    hough_lines = hough_transform(segmented_images, rho, theta, threshold, min_line_length, max_line_gap)
    img_with_lines = draw_lines(original_image, hough_lines)

    separated_lanes = separate_lines(hough_lines, original_image)
    img_different_lane_colors = color_lanes(original_image, separated_lanes[0], separated_lanes[1])
    return img_different_lane_colors
    full_lane_drawn_images = trace_both_lane_lines(original_image, separated_lanes[0], separated_lanes[1])

    show_image_list([ img_different_lane_colors,full_lane_drawn_images], cols=2, fig_size=(15, 15),
                    img_labels=['original', 'hsl'])

# Load the image
# original_image = cv2.imread('images2/solidWhiteRight.jpg')  # Replace 'your_lane_image.jpg' with the path to your image
# frame= process_lane_detection(original_image)
# cv2.imshow('Original Lane',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load Video
# Read Video file and processs
# cap = cv2.VideoCapture('training_data/video.mp4')
# subtracao = cv2.createBackgroundSubtractorMOG2()
# delay = 120
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     frame=process_lane_detection(frame)
#     cv2.imshow('Lane Detection',frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cap.release()
