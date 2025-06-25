import cv2
import numpy as np
import matplotlib as plt

def select_red(image):
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    return cv2.inRange(image, lower, upper)

def select_yellow(image):
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    return cv2.inRange(image, lower, upper)

def select_green(image):
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    return cv2.inRange(image, lower, upper)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255   
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def sliding_window_steering_angle(binary_warped, num_windows=12, margin=50, minpix=60, visualize=True):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Step 1: Base detection from histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // num_windows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_current = left_base
    right_current = right_base
    left_lane_inds = []
    right_lane_inds = []

    # Step 2: Sliding window tracking
    for window in range(num_windows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        if visualize:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Move window centers based on pixel mean
        if len(good_left_inds) > minpix:
            left_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = int(np.mean(nonzerox[good_right_inds]))

    # Step 3: Fit polynomial (degree 2 or 3)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 3) if len(leftx) > 100 else None
    right_fit = np.polyfit(righty, rightx, 3) if len(rightx) > 100 else None

    y_eval = binary_warped.shape[0] - 1
    if left_fit is not None and right_fit is not None:
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)
        lane_center = (left_x + right_x) / 2
    else:
        lane_center = (left_base + right_base) / 2

    image_center = binary_warped.shape[1] / 2
    dx = lane_center - image_center
    dy = y_eval
    steering_angle_rad = np.arctan2(dx, dy)
    steering_angle_deg = np.degrees(steering_angle_rad)

    # Step 4: Visualization
    if visualize:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        if left_fit is not None:
            left_fitx = np.polyval(left_fit, ploty)
            for i in range(len(ploty) - 1):
                cv2.line(out_img,
                         (int(left_fitx[i]), int(ploty[i])),
                         (int(left_fitx[i+1]), int(ploty[i+1])),
                         (255, 255, 0), 2)
        if right_fit is not None:
            right_fitx = np.polyval(right_fit, ploty)
            for i in range(len(ploty) - 1):
                cv2.line(out_img,
                         (int(right_fitx[i]), int(ploty[i])),
                         (int(right_fitx[i+1]), int(ploty[i+1])),
                         (255, 255, 0), 2)

        cv2.line(out_img, (int(lane_center), int(y_eval)), (int(lane_center), int(y_eval) - 50), (0, 255, 255), 2)
        cv2.line(out_img, (int(image_center), int(y_eval)), (int(image_center), int(y_eval) - 50), (255, 255, 255), 2)

        cv2.putText(out_img, f"Steering Angle: {steering_angle_deg:.2f} deg", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return steering_angle_deg, lane_center, image_center, out_img
    else:
        return steering_angle_deg, lane_center, image_center, None


#* Import the reusable functions
def create_hsv_trackbar(window_name="HSV Trackbars"):
    def nothing(x): pass
    cv2.namedWindow(window_name)
    cv2.createTrackbar("H Min", window_name, 0, 179, nothing)
    cv2.createTrackbar("S Min", window_name, 0, 255, nothing)
    cv2.createTrackbar("V Min", window_name, 0, 255, nothing)
    cv2.createTrackbar("H Max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V Max", window_name, 255, 255, nothing)

def get_hsv_values(window_name="HSV Trackbars"):
    h_min = cv2.getTrackbarPos("H Min", window_name)
    s_min = cv2.getTrackbarPos("S Min", window_name)
    v_min = cv2.getTrackbarPos("V Min", window_name)
    h_max = cv2.getTrackbarPos("H Max", window_name)
    s_max = cv2.getTrackbarPos("S Max", window_name)
    v_max = cv2.getTrackbarPos("V Max", window_name)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

def create_roi_trackbar(window_name="ROI Points"):
    def nothing(x): pass
    cv2.namedWindow(window_name)

    # Top Left
    cv2.createTrackbar("TL X", window_name, 0, 640, nothing)
    cv2.createTrackbar("TL Y", window_name, 0, 480, nothing)

    # Top Right
    cv2.createTrackbar("TR X", window_name, 640, 640, nothing)
    cv2.createTrackbar("TR Y", window_name, 0, 480, nothing)

    # Bottom Left
    cv2.createTrackbar("BL X", window_name, 0, 640, nothing)
    cv2.createTrackbar("BL Y", window_name, 480, 480, nothing)

    # Bottom Right
    cv2.createTrackbar("BR X", window_name, 640, 640, nothing)
    cv2.createTrackbar("BR Y", window_name, 480, 480, nothing)


def get_roi_points(window_name="ROI Points"):
    tl_x = cv2.getTrackbarPos("TL X", window_name)
    tl_y = cv2.getTrackbarPos("TL Y", window_name)
    tr_x = cv2.getTrackbarPos("TR X", window_name)
    tr_y = cv2.getTrackbarPos("TR Y", window_name)
    bl_x = cv2.getTrackbarPos("BL X", window_name)
    bl_y = cv2.getTrackbarPos("BL Y", window_name)
    br_x = cv2.getTrackbarPos("BR X", window_name)
    br_y = cv2.getTrackbarPos("BR Y", window_name)

    tl = [tl_x, tl_y]
    tr = [tr_x, tr_y]
    bl = [bl_x, bl_y]
    br = [br_x, br_y]

    return tl, tr, bl, br
