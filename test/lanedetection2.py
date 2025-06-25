import pyrealsense2 as rs
import numpy as np
import cv2

def nothing(x):
    pass

# Trackbars for HSV thresholding
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

prevLx = []
prevRx = []

# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    frame = cv2.resize(frame, (640, 480))

    # Perspective Points (adjust as needed)
    tl, bl, tr, br = (230, 75), (0, 430), (448, 78), (640, 380)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # HSV Thresholding
    hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    mask = cv2.inRange(hsv, (l_h, l_s, l_v), (u_h, u_s, u_v))

    # Histogram for base points
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window Parameters
    window_height = 30
    margin = 30
    minpix = 50
    y = 480
    lx, rx = [], []
    msk = mask.copy()

    while y > 0:
        # Left window
        left_x_low = left_base - margin
        left_x_high = left_base + margin
        img_left = mask[y-window_height:y, left_x_low:left_x_high]
        contours_left, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_left:
            if cv2.contourArea(contour) > minpix:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    lx.append(left_x_low + cx)
                    left_base = left_x_low + cx
                    break

        # Right window
        right_x_low = right_base - margin
        right_x_high = right_base + margin
        img_right = mask[y-window_height:y, right_x_low:right_x_high]
        contours_right, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_right:
            if cv2.contourArea(contour) > minpix:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    rx.append(right_x_low + cx)
                    right_base = right_x_low + cx
                    break

        # Visual debug
        cv2.rectangle(msk, (left_base - margin, y), (left_base + margin, y - window_height), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - margin, y), (right_base + margin, y - window_height), (255, 255, 255), 2)
        y -= window_height

    lx = lx if lx else prevLx
    rx = rx if rx else prevRx
    prevLx = lx
    prevRx = rx

    min_length = min(len(lx), len(rx))
    left_points = [(lx[i], 480 - i * window_height) for i in range(len(lx))]
    right_points = [(rx[i], 480 - i * window_height) for i in range(len(rx))]

    steering_angle = 0
    curvature = 0
    lane_offset = 0

    if len(lx) > 0 and len(rx) > 0:
        min_length = min(len(lx), len(rx))
        left_points = [(lx[i], 480 - i * window_height) for i in range(min_length)]
        right_points = [(rx[i], 480 - i * window_height) for i in range(min_length)]

        left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
        right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)

        y_eval = 480
        left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
        right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
        curvature = (left_curvature + right_curvature) / 2

        lane_center = (lx[0] + rx[0]) / 2
        car_position = 320
        lane_offset = (car_position - lane_center) * 3.7 / 640
        steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi
    elif len(lx) > 0 and len(rx) == 0:
        steering_angle = 45.0  # Turn right
    elif len(rx) > 0 and len(lx) == 0:
        steering_angle = -45.0  # Turn left
    else:
        steering_angle = 0

    if len(lx) > 0 and len(rx) > 0:
        top_left = (lx[0], 480)
        bottom_left = (lx[min_length-1], 0)
        top_right = (rx[0], 480)
        bottom_right = (rx[min_length-1], 0)
        quad_points = np.array([top_left, bottom_left, bottom_right, top_right], np.int32).reshape((-1, 1, 2))

        overlay = transformed_frame.copy()
        cv2.fillPoly(overlay, [quad_points], (0, 255, 0))
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    perspective_back = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))
    result = cv2.addWeighted(frame, 1, perspective_back, 0.5, 0)

    end_x = int(320 + 100 * np.sin(np.radians(steering_angle)))
    end_y = int(480 - 100 * np.cos(np.radians(steering_angle)))
    cv2.line(result, (320, 480), (end_x, end_y), (255, 0, 0), 2)

    cv2.putText(result, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if len(lx) == 0 or len(rx) == 0:
        cv2.putText(result, 'Partial Lane Detected', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show windows
    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Mask", mask)
    cv2.imshow("Lane Detection - Sliding", msk)
    cv2.imshow("Final Result", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
