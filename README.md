## Advanced Lane Finding 
### This file describes the steps for project 4 of the Udacity Self Driving Car Nanodegree
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Following files are provided:

* P4.ipynb - the main jupyter notebook
* lane_liny.py - Line class representing the lanes and is used in the jupyter notebook
* README.md - here you are
* several test and output images

[//]: # (Image References)

[undistorted]: ./output_images/chessboard_distort_undistort.png "Undistorted"
[test_undistorted]: ./output_images/test_distorted_undistorted.png "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[pipeline]: ./output_images/pipe.png "Pipeline"
[bird]: ./output_images/bird_eye_view.png "Bird-Eye View"
[transform]: ./output_images/color_sobel_transformation.png "Color/Sobel Transformation"
[mag_abs]: ./output_images/mag_abs.png "Mag/Abs Sobel Operation"
[result]: ./output_images/result.png "Result Image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3-5 code cells of the IPython notebook located in "P4.ipynb".  

Camera images often suffer from distorted images, like radial and tangential distortion. With the help of some calibration images we define the distortion coefficients which can be applied to undistort the images taken by this camera. They describe the relationshipt between the 3D point  coordinates and the position of this point on the image pixels. As calibration images chessboards are very suitable because of their known structure. 
I start by preparing "object points" (cell 3), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistorted]

### Pipeline (single images)
To get an short overview here you can see the applied pipeline after undistortion. The detailed steps are described underneath:

![Pipeline][pipeline]

#### 1. Provide an example of a distortion-corrected image.
I applied `cv2.undistort()` function on a real test image. Below you can see the distorted image on the left and the undistorted on the right. Especially on the edges of the image you can see the difference
![alt text][test_undistorted]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The function `get_perspective_matrices(img)` in cell 8 of the notebook takes an input image and calculates the distortion matrix and it's inverse. These two can be used as inputs for the function `warp()` for my perspective transform which is also defined in cell 8 to warp and get a bird-eye view and unwarp to get back to the normal view. The source and destination points are hardcoded and have the following values:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

Below a picture of warped image you can see that the lanes are nearly parallel.

![alt text][bird]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
After warping a combination of color and gradient thresholds are applied to the image to generate a binary image (cell 9-13). The color thresholds are pretty straightforward and are just thresholding the channels of some color spaces. The sobel is a derivative mask that can be applied in vertical and horizontal  direction and is used for edge detection. In our case to detect the lanes. As you can see in the notebook I played around with a lot of threshold functions and in the end I decided to apply the following:
* Color thresholds:
    - R channel of the RGB color space (222,255)
    - V channel of the HSV color space (220, 255)
    - L channel of the HLS color space (120, 250)
* Sobel thresholds:
    - Magnitude sobel on the V channel of the HSV color space (50, 200)
    - Magnitude sobel on the S channel of the HSV color space (80, 200)
Here you can see one absolute and magnitude sobel thresholds on the original test images:

![alt text][mag_abs]

All thresholds are combined via a bitwise OR (cell 12). Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][transform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In cell 17 of the notebook in the function `get_lane_base` we are making histogram based search for the lane base on the bottom half of the image by identifying the peaks. These results are the input for the `fit_sliding_window` (lane_line.py: 52) function of the `Line` class in file `lane_line.py`. Here we are sliding a small window from bottom to the top of the line. The mean of previous window valus is the center of the next window. Once the lane is detected the sliding window approach can be skipped and we can use the `fit_previous` function (lane_line.py: 112), where the fit of the previous lane and a margin is used to find the pixel of the new lane. With the detected pixels we calculate the second order polynom to fit the data. Overall the `Line` class which base was provided by course stores all properties of the lane and it's fit and alsp provides some the fit, drawing and sanity check functions.
Below you see the identified pixels and the fitted line of the second order polynom:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the class `Line` in `lane_line.py` on line 198 the function `calc_radius_of_curvature` is defined. It calculate the radius of the curve in a point by approximating a circle with the same tangent. [Here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) you find more information about this topic. The radius can be used for steering.

On line 214 the method `calc_line_base_pos` calculates the offset from the lane of the center. Like mentioned in the course I assume the lane to have a width of 3.7m and therefore the conversion from pixels to meter is set to 3.7/700 meter/pixel. Each lane calculates it's own offset from the center, by adding these two values the offset from the car from the center can easily determined.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 20 of the notebook in the function `process_image`. It executes the process described above on one image by using the `Line` class. Here is an example of my result on a test image:

![alt text][result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- hardcoded source destination points
- average left right lane
- take the derivative of the fit as sanity check

## References
[Udacity Repository](https://github.com/udacity/CarND-Advanced-Lane-Lines)

[The original writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)

[Radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)
