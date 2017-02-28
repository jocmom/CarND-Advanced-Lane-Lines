import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
Define a class to receive the characteristics of each line detection
"""
class Line():
    """
    Line class
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #x values for fitted line pixels
        self.fitx = None
        #y values for fitted line pixels
        self.fity = None

    def gen_fit(self, n_points):
        """
        Generate fit data for plot
        """
        self.fity = np.linspace(0, n_points-1, n_points)
        self.fitx = self.current_fit[0]*self.fity**2 + \
                    self.current_fit[1]*self.fity + \
                    self.current_fit[2]
        return self.fity, self.fitx

    def fit_sliding_window(self, binary_warped, base, nwindows=9, margin=100, minpix=40, plot=False):
        """
        Implementation of sliding window polynomial fit
        :param margin: Set the width of the windows +/-
        :param minpix: Set minimum number of pixels found to recenter window
        """
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Set height of windows
        window_height = int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        current = base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = current - margin
            win_x_high = current + margin
            # Draw the windows on the visualization image
            if plot:
                cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & \
                            (nonzeroy < win_y_high) & \
                            (nonzerox >= win_x_low) & \
                            (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]
        print(len(self.allx))

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

        if plot:
            # Generate x and y values for plotting
            self.gen_fit(binary_warped.shape[0])
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
            plt.imshow(out_img)
            plt.plot(self.fitx, self.fity, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()

        return self.current_fit

    def fit_previous(self, binary_warped, margin=100):
        """
        Assume you now have a new warped binary image
        from the next frame of video (also called "binary_warped")
        It's now much easier to find line pixels!
        """
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_inds = ((nonzerox > (self.current_fit[0]*(nonzeroy**2) + \
                                    self.current_fit[1]*nonzeroy + self.current_fit[2] - margin)) & \
                        (nonzerox < (self.current_fit[0]*(nonzeroy**2) + \
                                    self.current_fit[1]*nonzeroy + self.current_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]
        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        return self.current_fit

    def draw_poly(self, image, margin=100, plot=True):
        # Generate x and y values for plotting
        self.gen_fit(image.shape[0])
        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(image)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([self.fitx-margin, self.fity]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.fitx+margin, self.fity])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        image = cv2.addWeighted(image, 1, window_img, 0.3, 0)
        # Color in left and right line pixels
        self.draw_pixels(image)
        self.draw_fit(image)

        if plot:
            plt.imshow(image)
            #plt.plot(self.fitx, self.fity, color='yellow')
            plt.show()
        return image

    def draw_pixels(self, image):
        image[self.ally, self.allx] = [255, 0, 0]
        return image

    def draw_fit(self, image):
        line_fit = np.array([np.transpose(np.vstack([self.fitx, self.fity]))])
        cv2.polylines(image, np.int_([line_fit]), False, (255, 255, 0), 8)
        return image
    
    def draw_all(self, image):
        return image



