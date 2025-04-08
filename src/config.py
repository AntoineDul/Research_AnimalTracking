import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov11_pig_v2.pt")


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