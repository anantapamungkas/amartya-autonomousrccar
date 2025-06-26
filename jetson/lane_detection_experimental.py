import cv2
import numpy as np

# Initialize video capture
vidcap = cv2.VideoCapture("/home/juhdi/catkin_ws/src/amartya-autonomousrccar/jetson/data/Lane.mp4")
success, image = vidcap.read()

# Trackbars for HSV thresholding
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Store previous lane positions (for recovery if lanes are lost)
prevLx = [320] * 10  # Default left lane position (center-left)
prevRx = [420] * 10  # Default right lane position (center-right)
prev_left_fit = None
prev_right_fit = None

while success:
    success, image = vidcap.read()
    if not success:
        break  # Exit if video ends

    frame = cv2.resize(image, (640, 480))

    # Perspective transform points (bird's-eye view)
    tl = (222, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Apply perspective transform
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # HSV thresholding to detect lanes
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Histogram to find lane bases
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding window search
    y = 472
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        # Left lane window
        img = mask[y - 40:y, left_base - 50:left_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        # Right lane window
        img = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        # Draw sliding windows
        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    # Fallback to previous lane if current detection fails
    if len(lx) == 0:
        lx = prevLx
    else:
        prevLx = lx

    if len(rx) == 0:
        rx = prevRx
    else:
        prevRx = rx

    min_length = min(len(lx), len(rx))
    if min_length == 0:
        print("Warning: No lanes detected! Skipping frame.")
        continue  # Skip this frame

    # Generate lane points
    left_points = [(lx[i], 472 - i * 40) for i in range(min_length)]
    right_points = [(rx[i], 472 - i * 40) for i in range(min_length)]

    # Fit polynomials (with error handling)
    try:
        left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
        right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)
        prev_left_fit = left_fit
        prev_right_fit = right_fit
    except:
        print("Warning: Polynomial fit failed! Using previous fit.")
        if prev_left_fit is not None and prev_right_fit is not None:
            left_fit = prev_left_fit
            right_fit = prev_right_fit
        else:
            continue  # Skip if no previous fit available

    # Calculate curvature and offset
    y_eval = 480
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])
    curvature = (left_curvature + right_curvature) / 2

    lane_center = (left_base + right_base) / 2
    car_position = 320  # Assuming camera is centered
    lane_offset = (car_position - lane_center) * 3.7 / 640  # Convert to meters

    # Steering angle calculation
    steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi
    line_length = 100
    end_x = int(320 + line_length * np.sin(np.radians(steering_angle)))
    end_y = int(480 - line_length * np.cos(np.radians(steering_angle)))

    # Draw lane overlay
    top_left = (lx[0], 472)
    bottom_left = (lx[min_length - 1], 0)
    top_right = (rx[0], 472)
    bottom_right = (rx[min_length - 1], 0)
    quad_points = np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32).reshape((-1, 1, 2))
    overlay = transformed_frame.copy()
    cv2.fillPoly(overlay, [quad_points], (0, 255, 0))
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

    # Inverse perspective to original view
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    original_perspective_lane = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))
    result = cv2.addWeighted(frame, 1, original_perspective_lane, 0.5, 0)

    # Draw steering line and info
    cv2.line(result, (320, 480), (end_x, end_y), (255, 0, 0), 2)
    cv2.putText(result, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display all stages
    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Threshold", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)
    cv2.imshow("Lane Detection - Final", result)

    if cv2.waitKey(10) == 27:  # Exit on ESC
        break

vidcap.release()
cv2.destroyAllWindows()