import pyrealsense2 as rs
import numpy as np
import cv2
import math

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
    tl, bl, tr, br = (230, 75), (0, 430), (448, 78), (640, 380)
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
        return None  # No black region detected

# Convert x-error to steering angle
def calculate_steering_angle(error, y_offset=100):
    angle_rad = math.atan2(error, y_offset)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Main RealSense logic
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

            # Get frame data
            color_image = np.asanyarray(color_frame.get_data())

            # Mask and transform
            black_mask = get_black_mask_gray(color_image)
            bird_eye_mask = perspective_transform(black_mask)
            bird_eye_color = perspective_transform(color_image)

            # Find center of black region
            center = get_black_center(bird_eye_mask)
            h, w = bird_eye_mask.shape
            frame_center_x = w // 2

            if center:
                cx, cy = center
                error = cx - frame_center_x
                angle = calculate_steering_angle(error)
                servo_angle = map_angle_to_servo(angle)

                print(servo_angle)

                # Draw results
                cv2.circle(bird_eye_color, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(bird_eye_color, (frame_center_x, 0), (frame_center_x, h), (255, 255, 0), 2)
                cv2.putText(bird_eye_color, f"Angle: {angle:.2f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"[INFO] Angle: {angle:.2f} degrees")
            else:
                angle = 0
                print("[WARN] Black region not found!")

            # Display
            cv2.imshow("Warped View", bird_eye_color)
            cv2.imshow("Black Mask", bird_eye_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
