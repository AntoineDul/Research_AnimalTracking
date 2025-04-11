import os
from pathlib import Path

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov11_pig_v2.pt")
MEDIAFLUX_VIDEO_DIR = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-08-08-08\\Pen2_D5-8"
RFID_PATH = os.path.join(DATA_DIR, "RFID", "21-056 Drinker Raw Data 26Jun2024-18Sep2024.xlsx")


# Select detection model
DETECTION_METHOD = "YOLO"          # Options: "YOLO", "ColorSegmentation"

# Detection settings
YOLO_CONF_THRESHOLD = 0.65                      # Minimum confidence score for detection
YOLO_IMGSZ = 416                                # Image size expected by YOLO model 
LOWER_COLOR_RANGE = [0, 50, 50]                 # Lower bound for color detection in HSV space
UPPER_COLOR_RANGE = [30, 255, 255]              # Upper bound for color detection in HSV space
CONTOUR_AREA_THRESHOLD = 1000                   # Minimum area of contour to be considered a detection

# Select tracking method
TRACKING_METHOD = "GREEDY"                      # Options: "GREEDY", ("DEEP_SORT")

# Tracking settings
MAX_TRACK_AGE = 10                              # Max frames a lost object remains tracked
IOU_THRESHOLD = 0.6                             # Intersection over Union threshold for tracking
MAX_DISTANCE = 50                               # Maximum distance for matching detections to existing tracks   
ALPHA = 0.5                                     # Weight for IoU in cost function, weight of distance is (1 - alpha)

# Video processing settings
FRAME_SKIP = 2                                  # Number of frames to skip for processing

# Output settings
OUTPUT_VIDEO_WIDTH = 1600                        # Width of output video
OUTPUT_VIDEO_HEIGHT = 900                       # Height of output video
OUTPUT_VIDEO_FPS = 20                           # Frames per second for output video