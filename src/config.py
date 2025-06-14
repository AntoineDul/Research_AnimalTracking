import os
import numpy as np
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MEDIAFLUX_VIDEO_DIR = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08"   # Directory containing all videos to process

# Paths : files need to be created beforehand
TRACKING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "track_history")                       # Path to file where a json file with all tracks will be written every batch.
BATCH_PLOTS_PATH = os.path.join(OUTPUT_DIR, "batch_plots")                              # Path to file where all plots will be stored.
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov11_pig_v2.pt")                          # Path to fine-tuned yolo model weights.
LOGS_PATH = os.path.join(OUTPUT_DIR, f"logs\\{timestamp}_logs.txt")                     # logs file needs to be created.
CSV_PATH = os.path.join(DATA_DIR, "RFID", "manually_tracked_pigs.csv")                  # Path to ground truth positions at given timestamps

# Farm settings
NUM_PIGS = 15                                   # Total number of pigs in the pen
NUM_CAMERAS = 5                                 # Number of cameras in the pen 
FIRST_CAMERA = 5                                # First camera ID in the pen (pen 2)
CAM_ID_TO_CHANGE = {17: 9}                      # Dictionary with format {original_cam_id: new_cam_id} -> need the cam id to be sequential (e.g. 5, 6, 7, 8, 9)
RESOLUTION = (2592, 1944)                       # Camera resolution from official datasheet
DISTORTION = [-0.5, 0.1, 0, 0]                  # Distortion parameters used to undistort fisheye (found thorugh trial and error)
CAM_FULLY_OVERLAPPED = [5, 6, 7]                # List of cameras whose views are fully included in another camera view (no possibility of detecting a pig ONLY from these cams)
NON_OVERLAP_ZONES = [-1, 1]                     # Define the zones that can only be seen by one camera (e.g. [-1, 1] means that for all y coordinates below -1 or above 1, are points only seen in one pov)
CAM_POSITIONS = {                               # Approximated camera positions from video views -> to improve with more accurate measurements
    5: (-0.1, 0.5),
    6: (-0.1, -0.6),
    7: (1, 0),
    8: (0, -2.8),
    9: (2.2, 2.5)
    }
THALES_SCALE = {                                # Scale to undo bias of each cam (apply Thales theorem), these were approximated -> improve by replacing them by calculated constants from cam heights
    5: (0.94, 0.94),                
    6: (0.83, 0.95),
    7: (0.98, 0.93),
    8: (0.98, 0.93),
    9: (0.99, 0.93)
    }

# Select detection model and tracking method. NOTE: other methods could be implemented by following the base tracker or base detector abstract classes models
DETECTION_METHOD = "YOLO"                       # Options: "YOLO"
TRACKING_METHOD = "GREEDY"                      # Options: "GREEDY"

# Detection settings
YOLO_CONF_THRESHOLD = 0.65                      # Minimum confidence score for detection
YOLO_IMGSZ = 832                                # Image size expected by YOLO model 

# Tracking settings
MAX_TRACK_AGE = 30                              # Max frames a lost object remains tracked
IOU_THRESHOLD = 0.6                             # Intersection over Union threshold for tracking
MAX_DISTANCE = 50                               # Maximum distance for matching detections to existing tracks   
ALPHA = 0.5                                     # Weight for IoU in cost function, weight of distance is (1 - alpha)
MAX_PIG_MVMT_BETWEEN_TWO_FRAMES = 0.1           # max distance between two points in a batch to be considered from same paths and not outliers

# Batch analysis parameters
FRECHET_THRESHOLD = 1                           # Max frechet distance between 2 paths to be considered for merging
SIMILARITY_THRESHOLD = 0.05                     # Minimum distance for 2 paths to be considered distincts.
FRECHET_EUCLIDEAN_WEIGHTS = {                   # Weights for weighted sum for cost matrix when merging paths
    'Frechet': 0.5,     # 1
    'Euclidean': 1.5    # 0.5
    } 
BATCH_SIZE = 200                                # Number of frames processed before merging views together

# Video processing settings
REWIND_FRAMES = 20                              # Number of frames to rewind to overlap batches

# Output settings
OUTPUT_VIDEO_WIDTH = 1600                       # Width of output video
OUTPUT_VIDEO_HEIGHT = 900                       # Height of output video
OUTPUT_VIDEO_FPS = 50                           # Frames per second for output video

# Mapping settings for computing homography matrices 
# NOTE: this was tailored to batch 2 pen 2, would want a calibration sequence to replace this manually sampled data
MAPPINGS = {
    '5' : {
    'image_points': np.array(
        [[1580, 243], [1299, 203], [921, 145], [575, 93], [1096, 1699], [847, 1628], [611, 1546], [395, 1471], [90, 1385], [369, 62]], dtype=np.float32),
    'world_points': np.array(
        [[0, 0], [0.2, 0], [0.5, 0], [0.8, 0], [0.1, 1], [0.3, 1], [0.5, 1], [0.7, 1], [1, 1], [1, 0]], dtype=np.float32)
                 },
    '6' : {
        'image_points': np.array(
            [[1101, 18], [1524, 59], [1784, 79], [2135, 128], [2455, 160], [1239, 1564], [1619, 1538], [1972, 1509], [2283, 1493], [2575, 1465]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.3, 0], [0.5, 0], [0.8, 0], [1.1, 0], [0.1, -1], [0.4, -1], [0.7, -1], [1, -1], [1.3, -1]], dtype=np.float32)
                    },
    '7' : {
        'image_points': np.array(
            [[459, 881], [863, 889], [1269, 907], [1754, 929], [2075, 947], [463, 1727], [1003, 1753], [1546, 1777], [2041, 14], [1788, 4]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.5, 0], [1, 0], [1.6, 0], [2, 0], [0, -1], [0.7, -1], [1.4, -1], [1.9, 1], [1.6, 1]], dtype=np.float32)
                    },
    '8' : {
        'image_points': np.array(
            [[342, 1455], [382, 1113], [419, 766], [457, 481], [885, 1493], [916, 1220], [954, 883], [1012, 412], [25, 682], [54, 433]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.6, 0], [1.2, 0], [1.7, 0], [0.2, -1], [0.6, -1], [1.1, -1], [1.8, -1], [1.3, 1], [1.8, 1]], dtype=np.float32)
                    },
    '9' : {     # NOTE: camera 9 is actually camera 17 in the farm, we just use 9 for convenience
        'image_points': np.array(
            [[550, 372], [570, 744], [589, 1150], [609, 1449], [248, 557], [270, 820], [276, 1038],[293, 1287], [1008, 495], [1012, 813] ,[1031, 1120], [1041, 1483], [1709, 455], [1709, 861], [1717, 1326], [1715, 1700]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.7, 0], [1.5, 0], [2.1, 0], [0.2, -1], [0.8, -1] ,[1.3, -1], [1.9, -1], [0.4, 1], [0.9, 1], [1.4, 1], [2, 1], [0.5, 2], [1, 2], [1.6, 2], [2.1, 2]], dtype=np.float32)
                    },
    }
