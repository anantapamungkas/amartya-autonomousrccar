import pyrealsense2 as rs
import numpy as np
import cv2


#* Import the reusable functions
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

#* Create three HSV trackbar windows
create_hsv_trackbar("Green HSV")
create_hsv_trackbar("Yellow YUV")
create_hsv_trackbar("Red YUV")


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

    #* gausian blur
    blurFrame = cv2.GaussianBlur(color_image,(5,5),0)

    #* get trackbar value
    greenLower, greenUpper = get_hsv_values("Green HSV")
    yellowLower, yellowUpper = get_hsv_values("Yellow YUV")
    redLower, redUpper = get_hsv_values("Red YUV")

    #* green segmentation
    greenHSV = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(greenHSV, greenLower, greenUpper) 

    yellowHSV = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2YUV)
    yellowMask = cv2.inRange(yellowHSV, yellowLower, yellowUpper)
    
    redHSV = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2YUV)
    redMask = cv2.inRange(redHSV, redLower, redUpper)

    greenContour, _ = cv2.findContours(greenMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    greenMaxControur = max(greenContour, key=cv2.contourArea, default=None)

    yellowControur, _ = cv2.findContours(yellowMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellowMaxContour = max(yellowControur, key=cv2.contourArea, default=None)

    redContour, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redMaxContour = max(redContour, key=cv2.contourArea, default=None)

    if greenMaxControur is not None:
        [x, y, w, h] = cv2.boundingRect(greenMaxControur)
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        center_x = (x + (x + w)) // 2
        center_y = (y + (y + h)) // 2

        depthGreen = depth_frame.get_distance(center_x, center_y)
        cv2.line(color_image, (335,255),(center_x, center_y), (255,255,255), 2)
        cv2.putText(color_image, f"Depth: {depthGreen:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    if yellowMaxContour is not None:
        [x, y, w, h] = cv2.boundingRect(yellowMaxContour)
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        center_x = (x + (x + w)) // 2
        center_y = (y + (y + h)) // 2

        depthYellow = depth_frame.get_distance(center_x, center_y)
        cv2.line(color_image, (335,255),(center_x, center_y), (0,255,255), 2)
        cv2.putText(color_image, f"Depth: {depthYellow:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if redMaxContour is not None:
        [x, y, w, h] = cv2.boundingRect(redMaxContour)
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        center_x = (x + (x + w)) // 2
        center_y = (y + (y + h)) // 2

        depthRed = depth_frame.get_distance(center_x, center_y)
        cv2.line(color_image, (335,255),(center_x, center_y), (0,0,255), 2)
        cv2.putText(color_image, f"Depth: {depthRed:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Original", color_image)
    cv2.imshow("Green Mask", greenMask)
    cv2.imshow("Yellow Mask", yellowMask)
    cv2.imshow("Red Mask", redMask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
