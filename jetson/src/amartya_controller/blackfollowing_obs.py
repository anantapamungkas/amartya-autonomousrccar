import pyrealsense2 as rs
import numpy as np
import cv2
import math

# Mapping angle to servo pulse
def map_angle_to_servo(angle, angle_range=(-45, 45), servo_range=(60, 120)):
    angle = np.clip(angle, angle_range[0], angle_range[1])
    return np.interp(angle, angle_range, servo_range)

# Get black mask from grayscale using threshold + inversion
def get_black_mask_gray(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask

# Perspective warp
def perspective_transform(image):
    tl, bl, tr, br = (187, 173), (0, 480), (457, 170), (640, 444)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (640, 480))

# Get centroid of black region using image moments
def get_black_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return None

# Convert x-error to steering angle
def calculate_steering_angle(error, y_offset=100):
    angle_rad = math.atan2(error, y_offset)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# HSV trackbars
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

# Setup HSV windows
create_hsv_trackbar("Green HSV")
create_hsv_trackbar("Yellow HSV")
create_hsv_trackbar("Red HSV")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Obstacle distance threshold
OBSTACLE_THRESHOLD = 1.0  # in meters

# Obstacle detection helper
def get_obstacle_depth(mask, depth_frame):
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contour, key=cv2.contourArea, default=None)
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        depth = depth_frame.get_distance(cx, cy)
        return cx, depth, (x, y, w, h)
    return None, None, None

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get images
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Line-following preprocessing
        filtered = cv2.bilateralFilter(color_image, 9, 75, 75)
        color_blur = cv2.GaussianBlur(filtered, (3, 3), 0)
        black_mask = get_black_mask_gray(color_blur)
        bird_eye_mask = perspective_transform(black_mask)
        bird_eye_color = perspective_transform(color_blur)

        # Obstacle masks
        blurFrame = cv2.GaussianBlur(color_image, (5, 5), 0)
        # greenLower, greenUpper = get_hsv_values("Green HSV")
        # yellowLower, yellowUpper = get_hsv_values("Yellow HSV")
        # redLower, redUpper = get_hsv_values("Red HSV")

        greenLower = (32, 102, 66)
        greenUpper = (63, 255, 102)
        yellowLower = (91, 0, 161)
        yellowUpper = (179, 255, 255)
        redLower = (0, 0, 0)
        redUpper = (0, 0, 0)

        greenMask = cv2.inRange(cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV), greenLower, greenUpper)
        yellowMask = cv2.inRange(cv2.cvtColor(blurFrame, cv2.COLOR_BGR2YUV), yellowLower, yellowUpper)
        redMask = cv2.inRange(cv2.cvtColor(blurFrame, cv2.COLOR_BGR2YUV), redLower, redUpper)

        # Obstacle depth + direction
        def influence_by_obstacle(cx, depth, frame_center_x):
            if depth is not None and depth < OBSTACLE_THRESHOLD:
                if cx < frame_center_x:  # left side
                    return 30
                else:  # right side
                    return -30
            return 0

        cx_green, depth_green, box_g = get_obstacle_depth(greenMask, depth_frame)
        cx_yellow, depth_yellow, box_y = get_obstacle_depth(yellowMask, depth_frame)
        cx_red, depth_red, box_r = get_obstacle_depth(redMask, depth_frame)

        # Base angle from line-following
        center = get_black_center(bird_eye_mask)
        h, w = bird_eye_mask.shape
        frame_center_x = w // 2
        angle = 0

        if center:
            cx, cy = center
            error = cx - frame_center_x
            angle = calculate_steering_angle(error)

        # Obstacle influence
        angle += influence_by_obstacle(cx_green, depth_green, frame_center_x)
        angle += influence_by_obstacle(cx_yellow, depth_yellow, frame_center_x)
        angle += influence_by_obstacle(cx_red, depth_red, frame_center_x)

        servo_angle = map_angle_to_servo(angle)
        print(f"[INFO] Angle: {angle:.2f}°, Servo: {servo_angle:.2f}°")

        # Visualization
        if center:
            cv2.circle(bird_eye_color, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(bird_eye_color, (frame_center_x, 0), (frame_center_x, h), (255, 255, 0), 2)
        cv2.putText(bird_eye_color, f"Angle: {angle:.2f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if box_g:
            x, y, w, h = box_g
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        if box_y:
            x, y, w, h = box_y
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if box_r:
            x, y, w, h = box_r
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show results
        cv2.imshow("Warped View", bird_eye_color)
        cv2.imshow("Green Mask", greenMask)
        cv2.imshow("Yellow Mask", yellowMask)
        # cv2.imshow("Red Mask", redMask)
        cv2.imshow("Original", color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
