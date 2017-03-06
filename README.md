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
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3-5 code cells of the IPython notebook located in "P4.ipynb".  

Camera images often suffer from distorted images, like radial and tangential distortion. With the help of some calibration images we define the distortion coefficients which can be applied to undistort the images taken by this camera. As calibration images chessboards are very suitable because of their known structure.
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
After warping a combination of color and gradient thresholds are applied to the image to generate a binary image (cell 9-13). The color thresholds are pretty straightforward and are just thresholding the channels of some color spaces. The sobel thresholds . As you can see in the notebook I played around with a lot of threshold functions and in the end I decided to apply the following:
* Color thresholds:
    - R channel of the RGB color space (222,255)
    - V channel of the HSV color space (220, 255)
    - L channel of the HLS color space (120, 250)
* Sobel thresholds:
    - Magnitude sobel on the V channel of the HSV color space (50, 200)
    - Magnitude sobel on the S channel of the HSV color space (80, 200)

All thresholds are combined via a bitwise OR (cell 12). Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][transform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

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


