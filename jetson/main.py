import pyrealsense2 as rs
import matplotlib as plt
import numpy as np
import cv2
import utils

clicked_points = []

def get_mouse_click_coordinates(window_name="Image"):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            print(f"Clicked at: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    obs_angle = 0
    utils.create_hsv_trackbar("HSV")
    utils.create_roi_trackbar()

    tl = [230, 75]
    bl = [0, 430]
    tr = [448, 78]
    br = [640, 380]


    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        display_image = np.copy(color_image)   

        blur_image = utils.gaussian_blur(color_image, 3)

        filtered_color = utils.select_rgb_white_yellow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        blurred = utils.gaussian_blur(utils.canny(filtered_color, 100, 150), 7)
        rows, cols = color_image.shape[:2]
        
        # (x, y)
        bottom_left  = [int(cols*0.0), int(rows*1.0)]
        center_left  = [int(cols*0.0), int(rows*0.65)]
        top_left     = [int(cols*0.20), int(rows*0.55)]
        bottom_right = [int(cols*1), int(rows*1.0)]
        center_right = [int(cols*1), int(rows*0.65)]
        top_right    = [int(cols*0.75), int(rows*0.55)]
        vertices = np.array([[bottom_left, center_left, top_left, top_right, center_right, bottom_right]], dtype=np.int32)


        copied = np.copy(blurred)
        # cv2.line(copied, tuple(bottom_left), tuple(center_left), (255, 0, 0), 5)
        # cv2.line(copied, tuple(center_left), tuple(top_left), (255, 0, 0), 5)
        # cv2.line(copied, tuple(top_left), tuple(top_right), (255, 0, 0), 5)
        # cv2.line(copied, tuple(top_right), tuple(center_right), (255, 0, 0), 5)
        # cv2.line(copied, tuple(center_right), tuple(bottom_right), (255, 0, 0), 5)
        # cv2.line(copied, tuple(bottom_right), tuple(bottom_left), (255, 0, 0), 5)  # close the shape

        # ##* PERSPECTIVE TRANSFORMATION
        # src_pts = np.float32([
        #     [int(cols*0.23), int(rows*0.45)],
        #     [int(cols*0.90), int(rows*0.45)],
        #     [int(cols*1.0), int(rows*0.80)],
        #     [int(cols*0.0), int(rows*0.75)]
        # ])

        tl, tr, bl, br = utils.get_roi_points()

        # src_pts = np.float32([
        #     [175, 251],
        #     [473, 251],
        #     [638, 356],
        #     [1, 399],
        # ])

        src_pts = np.float32([
            tl,
            tr,
            br,
            bl,
        ])

        # Draw red points from src_pts on display_image
        pt0 = tuple(src_pts[0].astype(int))
        pt1 = tuple(src_pts[1].astype(int))
        pt2 = tuple(src_pts[2].astype(int))
        pt3 = tuple(src_pts[3].astype(int))

        cv2.circle(copied, pt0, 10, (255, 255, 255), -1)
        cv2.circle(copied, pt1, 10, (255, 255, 255), -1)
        cv2.circle(copied, pt2, 10, (255, 255, 255), -1)
        cv2.circle(copied, pt3, 10, (255, 255, 255), -1)

        cv2.imshow("Copied", copied)

        # cv2.circle(copied, tl, 5, (0,255,0), -1)
        # cv2.circle(copied, tr, 5, (0,255,0), -1)
        # cv2.circle(copied, bl, 5, (0,255,0), -1)
        # cv2.circle(copied, br, 5, (0,255,0), -1)

        # Destination points - form a rectangle
        dst_width = 640
        dst_height = 480
        dst_pts = np.float32([
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ])

        # Compute perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(color_image, M, (dst_width, dst_height))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        _, white_mask = cv2.threshold(equalized, 220, 255, cv2.THRESH_BINARY)
        copied = np.copy(white_mask)

        warped_gray = utils.grayscale(warped)
        warped_edges = utils.canny(warped_gray, 50, 150)
        warped_binary = utils.gaussian_blur(warped_edges, 5)

        cv2.imshow("warped", warped)

        steering_angle, lane_center, image_center, debug_img = utils.sliding_window_steering_angle(
            white_mask,
            visualize=True
        )

        #* obstacle segmentation
        greenMask = utils.select_green(cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV))
        yellowMask = utils.select_yellow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV))
        redMask = utils.select_red(cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV))

        greenContour, _ = cv2.findContours(greenMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        greenMaxControur = max(greenContour, key=cv2.contourArea, default=None)

        yellowControur, _ = cv2.findContours(yellowMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellowMaxContour = max(yellowControur, key=cv2.contourArea, default=None)

        redContour, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        redMaxContour = max(redContour, key=cv2.contourArea, default=None)

        if greenMaxControur is not None:
                [x, y, w, h] = cv2.boundingRect(greenMaxControur)
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                center_x = (x + (x + w)) // 2
                center_y = (y + (y + h)) // 2

                depthGreen = depth_frame.get_distance(center_x, center_y)

                # Calculate angle
                image_width = color_image.shape[1]
                image_center_x = image_width // 2
                HFOV = 69.4  # for Intel RealSense D435

                pixel_offset = center_x - image_center_x
                obs_angle = (pixel_offset / image_width) * HFOV

                cv2.line(display_image, (335,255),(center_x, center_y), (255,255,255), 2)
                cv2.putText(display_image, f"Depth: {depthGreen:.2f} m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            obs_angle = 0
             
                
        if yellowMaxContour is not None:
            [x, y, w, h] = cv2.boundingRect(yellowMaxContour)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            center_x = (x + (x + w)) // 2
            center_y = (y + (y + h)) // 2

            depthYellow = depth_frame.get_distance(center_x, center_y)

            # Calculate angle
            image_width = color_image.shape[1]
            image_center_x = image_width // 2
            HFOV = 69.4  # for Intel RealSense D435

            pixel_offset = center_x - image_center_x
            angle_offset_deg = (pixel_offset / image_width) * HFOV

            cv2.line(display_image, (335,255),(center_x, center_y), (0,255,255), 2)
            cv2.putText(display_image, f"Depth: {depthYellow:.2f} m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            obs_angle = 0

        if redMaxContour is not None:
            [x, y, w, h] = cv2.boundingRect(redMaxContour)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            center_x = (x + (x + w)) // 2
            center_y = (y + (y + h)) // 2

            depthRed = depth_frame.get_distance(center_x, center_y)

            # Calculate angle
            image_width = color_image.shape[1]
            image_center_x = image_width // 2
            HFOV = 69.4  # for Intel RealSense D435

            pixel_offset = center_x - image_center_x
            angle_offset_deg = (pixel_offset / image_width) * HFOV

            cv2.line(display_image, (335,255),(center_x, center_y), (0,0,255), 2)
            cv2.putText(display_image, f"Depth: {depthRed:.2f} m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            obs_angle = 0

        if debug_img is not None:
            cv2.imshow("Sliding Window Lane Detection", debug_img)

        print(f"Steering Angle: {steering_angle:.2f}°")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        get_mouse_click_coordinates("Copied")
        
        cv2.imshow("Display", display_image)

    cv2.destroyAllWindows()
    pipeline.stop()