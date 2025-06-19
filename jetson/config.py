import numpy as np

# HSV Value
LOWER_WHITE = np.array([0,0,230])
UPPER_WHITE = np.array([29,69,255])

LOWER_RED = np.array([0,66,234])
UPPER_REDT = np.array([28,109,255])

LOWER_YELLOW = np.array([87,0,149])
UPPER_YELLOW = np.array([109,137,187])

LOWER_GREEN = np.array([87,0,149])
UPPER_GREEN = np.array([109,137,187])

# region of interest point
TL = (248, 253)
BL = (13, 470)
TR = (427, 253)
BR = (631, 387)

# trackbar value settings
ROI_TRACKBAR = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

WHITE_TRACKBAR = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

RED_TRACKBAR = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

YELLOW_TRACKBAR = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

GREEN_TRACKBAR = [
    ("l_h", 0, 255), ("l_s", 0, 255), ("l_v", 200, 255),
    ("u_h", 255, 255), ("u_s", 50, 255), ("u_v", 255, 255)
]

