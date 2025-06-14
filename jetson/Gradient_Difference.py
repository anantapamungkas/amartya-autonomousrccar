import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def window():
    threshold1 = cv.getTrackbarPos("Threshold 1", "parameters")
    threshold2 = cv.getTrackbarPos("Threshold 2", "parameters")
    HoughTrigger = cv.getTrackbarPos("HoughTrigger", "parameters")
    HoughMaxGap = cv.getTrackbarPos("HoughMaxGap", "parameters")

    x1 = cv.getTrackbarPos("x1", "Roi")
    y1 = cv.getTrackbarPos("y1", "Roi")
    x2 = cv.getTrackbarPos("x2", "Roi")
    y2 = cv.getTrackbarPos("y2", "Roi")
    x3 = cv.getTrackbarPos("x3", "Roi")
    y3 = cv.getTrackbarPos("y3", "Roi")
    x4 = cv.getTrackbarPos("x4", "Roi")
    y4 = cv.getTrackbarPos("y4", "Roi")

    return threshold1, threshold2, HoughTrigger, HoughMaxGap,x1,y1 ,x2,y2,x3,y3,x4,y4,# Kembalikan nilai

def nothing(x):
    pass

def canny_edge_detection(image, threshold1=150, threshold2=200):
    """Detect edges in an image using Canny edge detection"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (3, 3))
    edges = cv.Canny(blur, threshold1, threshold2)
    return edges

def region_of_intrest(image,):
    points = np.array([[100, 100], [375, 100], [375, 400], [100, 400]], dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    mask = np.zeros_like(image)
    cv.fillPoly(mask, [points], color=(255, 255, 255))
    masked_image = cv.bitwise_and(image,mask)
    return masked_image,points

# Reading Videos
capture = cv.VideoCapture("/home/juhdi/catkin_ws/src/amartya-autonomousrccar/jetson/data/road.mp4")
_, frame = capture.read()

# Create window and trackbars
cv.namedWindow("parameters")
cv.createTrackbar("Threshold 1", "parameters", 150, 500, nothing)
cv.createTrackbar("Threshold 2", "parameters", 200, 500, nothing)
cv.createTrackbar("HoughTrigger", "parameters", 50, 500, nothing)
cv.createTrackbar("HoughMaxGap", "parameters", 50, 500, nothing)


while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    threshold1, threshold2, HoughTrigger, HoughMaxGap, polyX1, polyY1, polyX2, polyY2, polyX3, polyY3, polyX4, polyY4 = window()
    
    # Detect edges
    edges = canny_edge_detection(frame, threshold1, threshold2)
    crop_image,point = region_of_intrest(edges)
    
    # Detect lines
    lines = cv.HoughLinesP(edges, 1, np.pi/180, HoughTrigger, maxLineGap=HoughMaxGap)
    
    # Draw lines on original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Display results
    cv.imshow("Original with Lines", frame)
    cv.imshow('Edge Detection', crop_image)

    if cv.waitKey(30) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()