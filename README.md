# **Finding Lane Lines on the Road**

Identifying lanes on the road is a common task performed by all human drivers to ensure their vehicles are within lane constraints & with a safe distance from other vehicles on the road when driving, so as to make sure traffic is smooth and minimize chances of collisions with other cars due to lane misalignment.

Similarly, it is a critical task for an autonomous vehicle to perform. It turns out that recognizing lane markings on roads is possible using well known computer vision techniques. Some of those techniques will be covered below.


# Setup
YOLOv3 you need to download yolov3.weights file
[Download link](https://pjreddie.com/darknet/yolo/)


# **Functions**
 # Vehicle/Object Detection
 In this part, we will cover in detail the different steps needed to detect objects (Car,Truck,Road signs, Person, Animal, Cycle etc) on lane:
* Prerequisite : **yolov3.weights, yolov3.cfg and coco.names** files
* Input Image Information: The input image containing the scene with potential vehicles.
* Image Dimensions: Extracts the height and width of the input image. The third value _ is ignored as it typically corresponds to the number of color channels (e.g., 3 for RGB).
* Preprocessing: Prepares the image for input to a neural network. Scales, resizes the image dimensions and creates a blob that the neural network can process.
* Setting Input for the Neural Network: Sets the blob as the input to the neural network.
* Getting Output Layer Names: Retrieves the names of the output layers of the neural network.Performs a forward pass through the neural network to obtain the detections. The output is a list of detections.
* Postprocessing Detections: Extracts the confidence scores and class probabilities.Checks if the confidence score for the detected class is above a threshold, adds a text label and bounding box

 # Lane Detection
 
In this part, we will cover in detail the different steps needed to identify and classify lanes.
* Convert original image to HSL
* Isolate yellow and white from HSL image
* Combine isolated HSL with original image
* Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
* Convert image to grayscale for easier manipulation
* Apply Canny Edge Detection on smoothed gray image
* Perform a Hough Transform to find lanes within our region of interest and trace them in red
* Separate left and right lanes
* Interpolate line gradients to create two smooth lines

  # Solid/Dashed Lines Detection
 
In this part, we will cover in detail the different steps needed to identify and classify lane lines (Solid/Dashed).
* Convert original image to HSL
* Isolate yellow and white from HSL image
* Combine isolated HSL with original image
* Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
* Convert image to grayscale for easier manipulation
* Apply Canny Edge Detection on smoothed gray image
* Perform a Hough Transform to find lanes within our region of interest
* Classify Solid and dashed lines based on threshold length
* Draw both lines in different colors

# Future Improvements

* In the future, I also plan to apply Non-Maximum Suppression (NMS) to remove redundant bounding box predictions and improve the accuracy of object detection.
* Increase accuracy on the solid and dashed lines detection 
