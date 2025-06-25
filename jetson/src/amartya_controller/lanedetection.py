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

def get_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    return cv2.inRange(hsv, lower_white, upper_white)

def perspective_transform(image):
    h, w = image.shape[:2]
    src = np.float32([
        [230, 75],   # top-left
        [448, 78],   # top-right
        [640, 380],  # bottom-right
        [0, 430],    # bottom-left

    ])
    dst = np.float32([
        [w * 0.3, 0],
        [w * 0.7, 0],
        [w * 0.7, h],
        [w * 0.3, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))

def get_lane_position(mask):
    h, w = mask.shape
    roi_y = int(h * 0.85)
    roi = mask[roi_y:roi_y + 10, :]
    
    # Histogram across the ROI
    histogram = np.sum(roi, axis=0)
    
    midpoint = w // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    lane_center = (left_base + right_base) // 2
    frame_center = w // 2
    error = frame_center - lane_center
    
    return left_base, right_base, lane_center, frame_center, error

def calculate_steering_angle(error, y_offset=100):
    angle_rad = math.atan2(error, y_offset)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

if __name__ == "__main__":
    # while True:
    #     publish()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            white_mask = get_white_mask(color_image)
            bird_eye = perspective_transform(white_mask)
            color_vis = cv2.cvtColor(bird_eye, cv2.COLOR_GRAY2BGR)

            left_x, right_x, lane_center, frame_center, error = get_lane_position(bird_eye)
            angle = calculate_steering_angle(error)

            # Draw visualization
            h, w = bird_eye.shape
            roi_y = int(h * 0.85)
            cv2.line(color_vis, (left_x, roi_y), (left_x, roi_y + 10), (255, 0, 0), 2)
            cv2.line(color_vis, (right_x, roi_y), (right_x, roi_y + 10), (0, 255, 0), 2)
            cv2.line(color_vis, (lane_center, roi_y), (lane_center, roi_y + 10), (0, 0, 255), 2)
            cv2.line(color_vis, (frame_center, roi_y), (frame_center, roi_y + 10), (255, 255, 0), 1)

            cv2.putText(color_vis, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            print(f"Steering Angle: {angle:.2f} degrees")
            publish()

            cv2.imshow("Lane View", color_vis)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
