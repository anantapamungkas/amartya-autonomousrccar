# cap = cv2.VideoCapture('/home/juhdi/catkin_ws/src/my_robot_controller/scripts/road.mp4')  # or 0 for webcam

import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("hsv")

#
video = cv2.VideoCapture("/home/juhdi/catkin_ws/src/my_robot_controller/scripts/road.mp4")
cv2.createTrackbar("hue_min", "hsv", 0, 179, nothing)
cv2.createTrackbar("sat_min", "hsv", 0, 255, nothing)
cv2.createTrackbar("val_min", "hsv", 0, 255, nothing)
cv2.createTrackbar("hue_max", "hsv", 179, 179, nothing)
cv2.createTrackbar("sat_max", "hsv", 255, 255, nothing)
cv2.createTrackbar("val_max", "hsv", 255, 255, nothing)
cv2.createTrackbar("Min Area", "hsv", 500, 10000, nothing)

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("/home/juhdi/catkin_ws/src/my_robot_controller/scripts/road.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hue_min = cv2.getTrackbarPos("hue_min", "hsv")
    hue_max = cv2.getTrackbarPos("hue_max", "hsv")
    sat_min = cv2.getTrackbarPos("sat_min", "hsv")
    sat_max = cv2.getTrackbarPos("sat_max", "hsv")
    val_min = cv2.getTrackbarPos("val_min", "hsv")
    val_max = cv2.getTrackbarPos("val_max", "hsv")
    min_area = cv2.getTrackbarPos("Min Area", "hsv")

    low_yellow = np.array([hue_min, sat_min, val_min])
    up_yellow = np.array([hue_max, sat_max, val_max])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(20)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()