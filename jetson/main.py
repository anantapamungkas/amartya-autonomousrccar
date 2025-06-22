import pyrealsense2 as rs
import matplotlib as plt
import numpy as np
import cv2
import utils

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        filtered_color = utils.select_rgb_white_yellow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        blurred = utils.gaussian_blur(utils.canny(filtered_color, 100, 150), 7)
        rows, cols = color_image.shape[:2]
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

        ##* PERSPECTIVE TRANSFORMATION
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
        warped = cv2.warpPerspective(color_image, M, (dst_width, dst_height))

        warped_gray = utils.grayscale(warped)
        warped_edges = utils.canny(warped_gray, 50, 150)
        warped_binary = utils.gaussian_blur(warped_edges, 5)

        steering_angle, lane_center, image_center, debug_img = utils.sliding_window_steering_angle(
            warped_binary,
            visualize=True
        )

        if debug_img is not None:
            cv2.imshow("Sliding Window Lane Detection", debug_img)

        print(f"Steering Angle: {steering_angle:.2f}°")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    pipeline.stop()