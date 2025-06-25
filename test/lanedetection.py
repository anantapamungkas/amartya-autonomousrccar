import pyrealsense2 as rs
import numpy as np
import cv2
import math

def get_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    return cv2.inRange(hsv, lower_white, upper_white)

def get_white_mask_gray(frame, threshold=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

def perspective_transform(image):
    tl, bl, tr, br = (230, 75), (0, 430), (448, 78), (640, 380)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (640, 480))

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
            white_mask = get_white_mask_gray(color_image)
            bird_eye = perspective_transform(white_mask)
            bird_eye2 = perspective_transform(color_image)

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

            cv2.imshow("Warped", bird_eye2)
            cv2.imshow("Lane View", color_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
