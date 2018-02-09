import matplotlib.pyplot as plt

from camera_calibration import warp
from params import *


# detect lines
def find_first_frame(binary_warped):

    global ploty, car_pos, lane_dist
    #print("CAR_POS_1: ", car_pos)

    #print(binary_warped.shape)
    #print(binary_warped.dtype)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    # Find the peak (lane line) for left and right occurrences
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]) # X-indexes of nonzero elements
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height   # lower window border
        win_y_high = binary_warped.shape[0] - window*window_height      # higher border
        win_xleft_low = leftx_current - margin      # border left for the left window
        win_xleft_high = leftx_current + margin     # border right
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        # nonzero() returns two arrays of indexes of nonzero elements as a tupel(x,y)
        # .nonzero()[0] returns only X-indexes of nonzero elements
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If points within the window > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            # take the central (mean) index for the next window
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices of all points within the windows
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    Left.allx = nonzerox[left_lane_inds]
    Left.ally = nonzeroy[left_lane_inds]
    Right.allx = nonzerox[right_lane_inds]
    Right.ally = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each set of points
    # Calculate polynomial coefficients  a, b and c
    Left.current_fit = np.polyfit(Left.ally, Left.allx, 2)      # x and y are switched
    Right.current_fit = np.polyfit(Right.ally, Right.allx, 2)

    coeffs_left.append(Left.current_fit)
    coeffs_right.append(Right.current_fit)

    Left.current_fit = np.average(coeffs_left, axis=0)
    Right.current_fit = np.average(coeffs_right, axis=0)


    # smoothing the actual values and and validate the calculations


    #----------- Plot lane lines ------------------

    # Generate x and y values for plotting
    # Y-Values
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # Fit a second order polynomial to pixel positions for each lane line: x = ay^2 +by +c
    # X-Values
    left_fitx.append(Left.current_fit[0]*ploty**2 + Left.current_fit[1]*ploty + Left.current_fit[2])
    right_fitx.append(Right.current_fit[0]*ploty**2 + Right.current_fit[1]*ploty + Right.current_fit[2])

    # Plausibility checks
    # If the lane lines do not seem OK
    sanity_check()




    """
    dist = Right.line_base_pos - Left.line_base_pos
    if  (lane_dist*0.8 > dist) or (dist > lane_dist*1.2):
        Left.detected = False
        Right.detected = False
        print("Detected____________________________________")

    else:
        Left.detected = True
        Right.detected = True
    """

    print("FIRST")
    if show == True:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx[-1], ploty, color='yellow')
        plt.plot(right_fitx[-1], ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


def find_lines(binary_warped):
    print("SECOND")
    global ploty, car_pos



    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])     # X-indexes of nonzero elements
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_inds = ((nonzerox > (Left.current_fit[0] * (nonzeroy ** 2) + Left.current_fit[1] * nonzeroy + Left.current_fit[2] - margin)) &
                      (nonzerox < (Left.current_fit[0] * (nonzeroy ** 2) + Left.current_fit[1] * nonzeroy + Left.current_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (Right.current_fit[0] * (nonzeroy ** 2) + Right.current_fit[1] * nonzeroy + Right.current_fit[2] - margin)) &
                       (nonzerox < (Right.current_fit[0] * (nonzeroy ** 2) + Right.current_fit[1] * nonzeroy + Right.current_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    Left.allx = nonzerox[left_lane_inds]
    Left.ally = nonzeroy[left_lane_inds]
    Right.allx = nonzerox[right_lane_inds]
    Right.ally = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    Left.current_fit = np.polyfit(Left.ally, Left.allx, 2)
    Right.current_fit = np.polyfit(Right.ally, Right.allx, 2)
#
    coeffs_left.append(Left.current_fit)
    coeffs_right.append(Right.current_fit)

    Left.current_fit = np.average(coeffs_left, axis=0)
    Right.current_fit = np.average(coeffs_right, axis=0)


    # Generate x and y values for plotting
    # Fit a second order polynomial to pixel positions for each lane line: x = ay^2 +by +c
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])


    left_fitx.append(Left.current_fit[0] * ploty ** 2 + Left.current_fit[1] * ploty + Left.current_fit[2])
    right_fitx.append(Right.current_fit[0] * ploty ** 2 + Right.current_fit[1] * ploty + Right.current_fit[2])

    # Plausibility checks
    # If the lane lines do not seem OK use last values and search next lines with the find_first_frame() function.
    sanity_check()



    #----------- Plot lane lines ------------------
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx[-1] - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx[-1] + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx[-1] - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx[-1] + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))



    if show == True:
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.show()
        plt.plot(left_fitx[-1], ploty, color='yellow')
        plt.plot(right_fitx[-1], ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)





def get_dist_radius():
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # x= mx/my^2 * ay^2 + mx/my * by+c


    # Fit new polynomials to x,y in world space
    Left.allx = Left.allx [::-1]  # Reverse to match top-to-bottom in y
    Right.allx = Right.allx[::-1]  # Reverse to match top-to-bottom in y
    left_fit_cr = np.polyfit(Left.ally * ym_per_pix, Left.allx * xm_per_pix, 2)
    #left_fit_cr = np.polyfit(ploty * ym_per_pix, Left.allx * xm_per_pix, 2)

    right_fit_cr = np.polyfit(Right.ally * ym_per_pix, Right.allx * xm_per_pix, 2)
    #right_fit_cr = np.polyfit(ploty * ym_per_pix, Right.allx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    Left.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
                               / np.absolute(2 * left_fit_cr[0])
    Right.radius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
                                / np.absolute(2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    print("Curvature: ", Left.radius_of_curvature, 'm', Right.radius_of_curvature, 'm')


def draw_lines(warped, undist):
    global car_pos
    print("CAR_POS_5: ", car_pos)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx[-1], ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[-1], ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
#    newwarp = cv2.warpPerspective(color_warp, Minv, (size_y, size_x))
    newwarp = warp(color_warp, dst, src)
    #print("SHAPES: ", undist.shape, newwarp.shape)
    # Combine the result with the original image

    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


    radiuses.append(int((Left.radius_of_curvature + Right.radius_of_curvature) / 2))
    radius_of_curvature = int(np.average(radiuses))


    car_pos = Left.line_base_pos + (Right.line_base_pos - Left.line_base_pos) / 2
    distance_from_center = abs(640-car_pos) * xm_per_pix

    print("CAR_POS_6: ", car_pos)

    # Print distance from center on video

    if car_pos > 640:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (10, 80),
                    fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (10, 80),
                    fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {} m'.format(radius_of_curvature), (10, 120),
                fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.putText(result, 'Line Distance: {}m'.format((Right.line_base_pos - Left.line_base_pos)* xm_per_pix), (0, 240), fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(result, 'Left line in pxl {}pxl'.format(Left.line_base_pos), (10, 160), fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(result, 'Right line in pxl {}pxl'.format(Right.line_base_pos), (10, 200), fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)

    return result

def sanity_check():
    global ploty, error_counter
    error_flag = False

    print("\n--------------------------- START ------------------------------")


    if len(left_fitx[-1]) > 0:
        distance = (np.asarray(right_fitx) - np.asarray(left_fitx[-1]))
        distance_bottom = (right_fitx[-1][size_y-1] - left_fitx[-1][size_y-1])
        d_max = np.amax(distance)
        d_min = np.amin(distance)
    else:
        d_max = 0
        d_min = 0
        distance = 0
        distance_bottom = 0


    # Save coefficient differences
    if (len(coeffs_left) > 1) and (len(coeffs_right) > 1):
        Left.diffs = coeffs_left[-1] - coeffs_left[-2]
        Right.diffs = coeffs_right[-1] - coeffs_right[-2]


    # Sanity Check
    if len(left_fitx) > 1:

        # test for the line displacement
        # If one of the lines are displaced for more than 50 pixels (approx. 0.3m) set line base position and coefficients to last values
        # Left Line
        if abs(Left.line_base_pos - left_fitx[-1][size_y-1]) > line_deviation_tollerance:
            coeffs_left[-1]= coeffs_left[-2]
            left_fitx.append(left_fitx[-2])
            Left.detected = False
            error_flag = True
        else:
            Left.detected = True


        # Right Line
        if abs(Right.line_base_pos - right_fitx[-1][size_y-1]) > line_deviation_tollerance:
            coeffs_right[-1]= coeffs_right[-2]
            right_fitx.append(right_fitx[-2])
            Right.detected = False
            error_flag = True
        else:
            Right.detected = True



        if error_flag == True:
            error_counter += 1
        else:
            if error_counter > 0:
                error_counter -= 1

        if error_counter > 4:
            left_fitx.fill(0)
            right_fitx.fill(0)



        # If the lines are not parallel within a tolerance of approx. 10%, or
        # the line width exceed the max value (supposed line width + 10% tolerance)

        if ((d_min * 1.1) < d_max) or (d_max > lane_dist * 1.1):
            print("____________________NOT_ PARALLEL")
            Left.detected = False
            Right.detected = False

        print("\n--------------------------- STOP ------------------------------")

        Left.current_fit = coeffs_left[-1]
        Left.current_fit = coeffs_left[-1]
        Right.line_base_pos = right_fitx[-1][size_y - 1]
        Left.line_base_pos = left_fitx[-1][size_y - 1]


