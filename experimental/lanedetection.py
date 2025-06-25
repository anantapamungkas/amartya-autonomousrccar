import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

import math


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

def sliding_window_steering_angle(binary_warped, num_windows=9, margin=50, minpix=50, visualize=True):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
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

        if len(good_left_inds) > minpix:
            left_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

    y_eval = binary_warped.shape[0] - 1
    if left_fit is not None and right_fit is not None:
        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2
    else:
        lane_center = (left_base + right_base) / 2

    image_center = binary_warped.shape[1] / 2
    dx = lane_center - image_center
    dy = y_eval
    steering_angle_rad = np.arctan2(dx, dy)
    steering_angle_deg = np.degrees(steering_angle_rad)

    if visualize:
        # Draw lane pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw fitted polynomials
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            for i in range(len(ploty)-1):
                cv2.line(out_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255,255,0), 2)
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            for i in range(len(ploty)-1):
                cv2.line(out_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (255,255,0), 2)

        # Draw lane center and image center
        cv2.line(out_img, (int(lane_center), int(y_eval)), (int(lane_center), int(y_eval)-50), (0, 255, 255), 2)
        cv2.line(out_img, (int(image_center), int(y_eval)), (int(image_center), int(y_eval)-50), (255, 255, 255), 2)
        cv2.putText(out_img, f"Steering Angle: {steering_angle_deg:.2f} deg", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        plt.figure(figsize=(8,10))
        plt.imshow(out_img.astype(np.uint8))
        plt.title("Sliding Window Visualization")
        plt.axis("off")
        plt.show()

    return steering_angle_deg, lane_center, image_center


image = cv2.imread('resources\image.png')
filtered_color = select_rgb_white_yellow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

blurred = gaussian_blur(canny(filtered_color, 100, 150), 7)
rows, cols = image.shape[:2]
# (x, y)
bottom_left  = [int(cols*0.0), int(rows*0.95)]
center_left  = [int(cols*0.0), int(rows*0.30)]
top_left     = [int(cols*0.20), int(rows*0.20)]
bottom_right = [int(cols*1), int(rows*0.95)]
center_right = [int(cols*1), int(rows*0.40)]
top_right    = [int(cols*0.75), int(rows*0.20)]
vertices = np.array([[bottom_left, center_left, top_left, top_right, center_right, bottom_right]], dtype=np.int32)

copied = np.copy(blurred)
cv2.line(copied, tuple(bottom_left), tuple(center_left), (255, 0, 0), 5)
cv2.line(copied, tuple(center_left), tuple(top_left), (255, 0, 0), 5)
cv2.line(copied, tuple(top_left), tuple(top_right), (255, 0, 0), 5)
cv2.line(copied, tuple(top_right), tuple(center_right), (255, 0, 0), 5)
cv2.line(copied, tuple(center_right), tuple(bottom_right), (255, 0, 0), 5)
cv2.line(copied, tuple(bottom_right), tuple(bottom_left), (255, 0, 0), 5)  # close the shape

plt.imshow(region_of_interest(copied,vertices))
plt.show()

### ✅ PERSPECTIVE TRANSFORMATION

# Select 4 key points from the 6 to define a trapezoid
src_pts = np.float32([
    [635, 190],
    [960, 190],
    [1283,500],
    [320, 500]
])

# Destination points - form a rectangle
dst_width = 400
dst_height = 600
dst_pts = np.float32([
    [0, 0],
    [dst_width, 0],
    [dst_width, dst_height],
    [0, dst_height]
])

# Compute perspective transform matrix and apply it
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(image, M, (dst_width, dst_height))

# Show the result
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Perspective Transform (Bird's Eye View)")
plt.axis("off")
plt.show()

# Assume you already have this from earlier
warped_gray = grayscale(warped)
warped_edges = canny(warped_gray, 50, 150)
warped_binary = gaussian_blur(warped_edges, 5)

# Option 1: Show result with plot (for testing/debugging)
steering_angle, lane_center, image_center = sliding_window_steering_angle(
    warped_binary,
    visualize=False  # 👈 toggle this
)

print(f"Steering Angle: {steering_angle:.2f}°")