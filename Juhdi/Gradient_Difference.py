import cv2 as cv
import numpy as np

def nothing(x):
    pass

def canny_edge_detection(image, threshold1=150, threshold2=200):
    """Detect edges in an image using Canny edge detection"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (3, 3))
    edges = cv.Canny(blur, threshold1, threshold2)
    return edges

# Create window and trackbars
cv.namedWindow("parameters")
cv.createTrackbar("Threshold 1", "parameters", 150, 500, nothing)
cv.createTrackbar("Threshold 2", "parameters", 200, 500, nothing)
cv.createTrackbar("HoughTrigger", "parameters", 50, 500, nothing)
cv.createTrackbar("HoughMaxGap", "parameters", 50, 500, nothing)

# Reading Videos
capture = cv.VideoCapture("/home/juhdi/catkin_ws/src/amartya-autonomousrccar/Juhdi/src/road.mp4")

while True:
    isTrue, frame = capture.read()
    
    if not isTrue:
        break
        
    # Get current trackbar positions
    threshold1 = cv.getTrackbarPos("Threshold 1", "parameters")
    threshold2 = cv.getTrackbarPos("Threshold 2", "parameters")
    HoughTrigger = cv.getTrackbarPos("HoughTrigger", "parameters")
    HoughMaxGap = cv.getTrackbarPos("HoughMaxGap", "parameters")
    
    
    # Detect edges
    edges = canny_edge_detection(frame, threshold1, threshold2)
    
    # Detect lines
    lines = cv.HoughLinesP(edges, 1, np.pi/180, HoughTrigger, maxLineGap=HoughMaxGap)
    
    # Draw lines on original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Display results
    cv.imshow("Original with Lines", frame)
    cv.imshow('Edge Detection', edges)
    
    if cv.waitKey(30) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()