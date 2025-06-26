import pyrealsense2 as rs
import numpy as np
import cv2
import math
import rospy
from geometry_msgs.msg import Twist

def publish():
    rospy.init_node('cmd_vel_publisher', anonymous=True)     # Initialize the node
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)    # Create a publisher object
    rate = rospy.Rate(10)  # 10 Hz

    twist = Twist()
    twist.linear.x = 0   # Forward with 0.5 m/s
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = angle  # Rotate with 0.5 rad/s

    # Publish in a loop
    while not rospy.is_shutdown():
        pub.publish(twist)
        rospy.loginfo(f"Publishing cmd_vel: linear.x={twist.linear.x}, angular.z={twist.angular.z}")
        rate.sleep()

# === Mapping angle to servo pulse ===
def map_angle_to_servo(angle, angle_range=(-45, 45), servo_range=(60, 120)):
    angle = np.clip(angle, angle_range[0], angle_range[1])
    return np.interp(angle, angle_range, servo_range)

# === Black line detection ===
def get_black_mask_gray(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask

# === Perspective transform for line following ===
def perspective_transform(image):
    tl, bl, tr, br = (187, 173), (0, 480), (457, 170), (640, 444)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (640, 480))

# === Find center of black region ===
def get_black_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return None

# === Error to steering angle ===
def calculate_steering_angle(error, y_offset=100):
    angle_rad = math.atan2(error, y_offset)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# === Obstacle detection from color frame ===
def get_largest_obstacle(mask, depth_frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea, default=None)
    if max_contour is not None and cv2.contourArea(max_contour) > 300:
        x, y, w, h = cv2.boundingRect(max_contour)
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        depth = depth_frame.get_distance(cx, cy)
        return cx, cy, depth, (x, y, w, h)
    return None, None, None, None

# === Obstacle avoidance influence ===
def influence_by_obstacle(cx, depth, frame_center_x, threshold=1.0):
    if depth is not None and depth < threshold:
        if cx < frame_center_x:
            return 30
        else:
            return -30
    return 0

# === RealSense init ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === Distance threshold for avoidance ===
OBSTACLE_THRESHOLD = 1.0  # meters

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # === Black line preprocessing ===
        filtered = cv2.bilateralFilter(color_image, 9, 75, 75)
        color_blur = cv2.GaussianBlur(filtered, (3, 3), 0)
        black_mask = get_black_mask_gray(color_blur)
        bird_eye_mask = perspective_transform(black_mask)
        bird_eye_color = perspective_transform(color_blur)

        # === Obstacle color masks ===
        blurFrame = cv2.GaussianBlur(color_image, (5, 5), 0)
        hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2YUV)

        # Define HSV/YUV ranges
        greenLower = (32, 102, 66)
        greenUpper = (63, 255, 102)
        yellowLower = (91, 0, 161)
        yellowUpper = (179, 255, 255)
        redLower = (0, 0, 0)
        redUpper = (0, 0, 0)

        greenMask = cv2.inRange(hsv, greenLower, greenUpper)
        yellowMask = cv2.inRange(yuv, yellowLower, yellowUpper)
        redMask = cv2.inRange(yuv, redLower, redUpper)

        # === Combine all obstacle masks ===
        combined_mask = cv2.bitwise_or(greenMask, yellowMask)
        combined_mask = cv2.bitwise_or(combined_mask, redMask)
        cx_obs, cy_obs, depth_obs, box_obs = get_largest_obstacle(combined_mask, depth_frame)

        # === Base angle from line ===
        center = get_black_center(bird_eye_mask)
        h, w = bird_eye_mask.shape
        frame_center_x = w // 2
        angle = 0

        if center:
            cx, cy = center
            error = cx - frame_center_x
            angle = calculate_steering_angle(error)

        # === Add obstacle influence ===
        angle += influence_by_obstacle(cx_obs, depth_obs, frame_center_x, OBSTACLE_THRESHOLD)

        # === Servo output ===
        servo_angle = map_angle_to_servo(angle)
        depth_str = f"{depth_obs:.2f} m" if depth_obs is not None else "N/A"
        print(f"[INFO] Angle: {angle:.2f}°, Servo: {servo_angle:.2f}°, Depth: {depth_str}")

        # === Visualize line-following ===
        if center:
            cv2.circle(bird_eye_color, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(bird_eye_color, (frame_center_x, 0), (frame_center_x, h), (255, 255, 0), 2)
        cv2.putText(bird_eye_color, f"Angle: {angle:.2f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === Visualize obstacle ===
        if box_obs:
            x, y, w, h = box_obs
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            if depth_obs is not None:
                cv2.putText(color_image, f"{depth_obs:.2f} m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # === Show windows ===
        cv2.imshow("Warped View", bird_eye_color)
        cv2.imshow("Combined Obstacle Mask", combined_mask)
        cv2.imshow("Original", color_image)

        publish()

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
