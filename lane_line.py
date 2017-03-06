
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
    #maximum distance from lane to center in meters
    MAX_DISTANCE_TO_BASE = 2.2
    #minimum distance from lane to center in meters
    MIN_DISTANCE_TO_BASE = 1.0
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

    def gen_fit(self, n_points, fit):
        """
        Generate fit data for plot
        """
        fity = np.linspace(0, n_points-1, n_points)
        fitx = fit[0] * fity**2 + \
               fit[1] * fity + \
               fit[2]
        return fitx, fity

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
        allx = nonzerox[lane_inds]
        ally = nonzeroy[lane_inds]
        #print("Number of data points", len(allx))
        self.is_lane_valid(binary_warped, allx, ally)

        if plot:
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
        lane_inds = ((nonzerox > (self.best_fit[0]*(nonzeroy**2) + \
                                    self.best_fit[1]*nonzeroy + self.best_fit[2] - margin)) & \
                        (nonzerox < (self.best_fit[0]*(nonzeroy**2) + \
                                    self.best_fit[1]*nonzeroy + self.best_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        allx = nonzerox[lane_inds]
        ally = nonzeroy[lane_inds]
        self.is_lane_valid(binary_warped, allx, ally)
        return self.best_fit

    def is_lane_valid(self, binary_warped, allx, ally):
        """
        Check wheter lane is valid and if so calculate best fit
        """
        if len(allx) < 100:
            print("Not enough data points")
            return False

        # Fit a second order polynomial to each
        self.allx = allx
        self.ally = ally
        self.current_fit = np.polyfit(ally, allx, 2)
        if self.detected:
            if self.is_fit_diff_valid() == False:
                return False
        else:
            self.best_fit = self.current_fit
        # Generate x and y values for plotting
        self.fitx, self.fity = self.gen_fit(binary_warped.shape[0], self.best_fit)
        self.radius_of_curvature = self.calc_radius_of_curvature()
        self.line_base_pos = self.calc_line_base_pos(binary_warped)
        if self.is_base_pos_valid() == False: 
            return False
        self.average_fit()
        #self.best_fit = self.current_fit
        self.fitx, self.fity = self.gen_fit(binary_warped.shape[0], self.best_fit)
        self.detected = True
        return True

    def is_fitx_valid(self, binary_warped, fitx):
        """
        Sanity check to detect whether origin is outside of image
        """
        if fitx[-1] < 0 or fitx[-1] > binary_warped.shape[1]:
            print("Sanity Check: Origin not valid")
            return False
        return True

    def is_fit_diff_valid(self):
        """
        Sanity check with comparison of the polynomial coefficients from best fit 
        """
        self.diffs = np.absolute(self.current_fit - self.best_fit)
        if self.diffs[1] > 0.5 or self.diffs[0] > 0.0005:
        #if self.diffs[1] > 1 or self.diffs[0] > 0.001:
            print("Sanity Check: Difference to best fit too big", self.diffs)
            return False
        return True

    def is_base_pos_valid(self):
        """
        Sanity check: Is the base position of the car correct
        """
        if np.absolute(self.line_base_pos) > self.MAX_DISTANCE_TO_BASE or \
           np.absolute(self.line_base_pos) < self.MIN_DISTANCE_TO_BASE:
            print("Sanity Check: Invalid base position", self.line_base_pos)
            return False
        return True

    def average_fit(self, factor=0.1):
        """
        Average best fit with current fit
        """
        self.best_fit = self.best_fit * (1-factor) + self.current_fit * factor
        return self.best_fit

    def calc_radius_of_curvature(self):
        """
        Calculate curvature radius in meters
        """
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.fity)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.fity * ym_per_pix, self.fitx * xm_per_pix, 2)
        # Calculate the new radius of curvature
        radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / \
                                   np.absolute(2 * fit_cr[0])
        return radius_of_curvature

    def calc_line_base_pos(self, image):
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        image_center = int(image.shape[1]/2)
        # line base is last element of x values
        self.line_base_pos = image_center - self.fitx[-1]
        self.line_base_pos *= xm_per_pix
        return self.line_base_pos

    def draw_poly(self, image, margin=100, plot=False):
        """
        Draw Polygon on image, must be an image no binary
        """
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

        if plot:
            plt.imshow(image)
            #plt.plot(self.fitx, self.fity, color='yellow')
            plt.show()
        return image

    def draw_pixels(self, image):
        """
        Draw detected lane pixels
        """
        image[self.ally, self.allx] = [255, 0, 0]
        return image

    def draw_fit(self, image):
        """
        Draw calculated fit
        """
        line_fit = np.array([np.transpose(np.vstack([self.fitx, self.fity]))])
        cv2.polylines(image, np.int_([line_fit]), False, (50, 255, 255), 16)
        return image

    def draw_curvature(self, image, pos=(80,100)):
        """
        Print curvature text
        """
        #text = "Curvature: " + str(self.radius_of_curvature)
        text = '{}{:+6.0f}{}'.format("Curvature: ", self.radius_of_curvature, " m")
        return cv2.putText(image, text, pos, fontFace=cv2.FONT_HERSHEY_COMPLEX, \
                           fontScale=1, color=(255, 255, 255), thickness=2)

    def draw_center_offset(self, image, pos=(740,100)):
        """
        Print Offset from lane center text
        """
        text = '{}{:+4.2f}{}'.format("Center offset: ", self.line_base_pos, " m")
        return cv2.putText(image, text, pos, fontFace=cv2.FONT_HERSHEY_COMPLEX, \
                           fontScale=1, color=(255, 255, 255), thickness=2)

    def draw_all(self, image):
        """
        Draw all kind of stuff
        """
        #image = self.draw_poly(image)
        # Color in left and right line pixels
        image = self.draw_pixels(image)
        image = self.draw_fit(image)
        return image



