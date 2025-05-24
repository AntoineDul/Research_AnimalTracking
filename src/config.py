import os
import numpy as np

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MEDIAFLUX_VIDEO_DIR = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08"

# Paths
TRACKING_HISTORY_PATH = os.path.join(OUTPUT_DIR, "track_history\\tracking_history.json")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "track_history\\global_movements.png")
BATCH_PLOTS_PATH = os.path.join(OUTPUT_DIR, "batch_plots")
PROCESSED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "processed_videos\\tracked_pigs.avi")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov11_pig_v2.pt")
RFID_PATH = os.path.join(DATA_DIR, "RFID", "21-056 Drinker Raw Data 26Jun2024-18Sep2024.xlsx")

# Farm settings
NUM_PIGS = 20    #15                            # Total number of pigs in the pen
NUM_CAMERAS = 5                                 # Number of cameras in the pen monitored
FIRST_CAMERA = 5                                # First camera ID in the pen monitored (pen 2)
RESOLUTION = (2592, 1944)                       # Camera resolution from official datasheet
DISTORTION = [-0.5, 0.1, 0, 0]                  # Distortion parameters used to undistort fisheye (found thorugh trial and error)
CAM_FULLY_OVERLAPPED = [5, 6, 7]                # List of cameras whose views are fully included in another camera view (no possibility of detecting a pig ONLY from these cams)
CAM_BIAS = {                                    # Bias (k1, k2) of cameras, with the format (x, y) = (k1 * x, k2 * y)
    5: [
        {"det_center": (377, 497), "true_footprint": (463, 523)},
        {"det_center": (943, 244), "true_footprint": (977, 307)},
        {"det_center": (1397, 984), "true_footprint": (1445, 1029)},
        {"det_center": (988, 1398), "true_footprint": (999, 1368)},
    ],
    6: [
        {"det_center": (1526, 1234), "true_footprint": (1382, 1216)},
        {"det_center": (1868, 1710), "true_footprint": (1735, 1689)},
        {"det_center": (2516, 1415), "true_footprint": (2303, 1432)},
        {"det_center": (1609, 386), "true_footprint": (1514, 416)},
        {"det_center": (1405, 1696), "true_footprint": (1357, 1578)},
    ],
    7: [
        {"det_center": (1654, 1736), "true_footprint": (1626, 1686)},
        {"det_center": (1943, 1017), "true_footprint": (1920, 1000)},
        {"det_center": (1993, 311), "true_footprint": (1943, 395)},
        {"det_center": (609, 885), "true_footprint": (630, 870)},
        {"det_center": (521, 1812), "true_footprint": (564, 1736)},
        {"det_center": (2041, 1690), "true_footprint": (1962, 1618)},
    ],
    8: [
        {"det_center": (2024, 1358), "true_footprint": (2025, 1359)},
        {"det_center": (634, 1025), "true_footprint": (712, 1100)},
        {"det_center": (1260, 683), "true_footprint": (1322, 748)},
        {"det_center": (911, 226), "true_footprint": (967, 353)},
        {"det_center": (608, 252), "true_footprint": (686, 383)},
        {"det_center": (536, 1476), "true_footprint": (630, 1492)},
    ],
    9: [
        {"det_center": (2144, 1010), "true_footprint": (2144, 1012)},
        {"det_center": (2133, 1512), "true_footprint": (2133, 1512)},
        {"det_center": (2189, 434), "true_footprint": (2189, 438)},
        {"det_center": (1058, 903), "true_footprint": (1144, 944)},
        {"det_center": (460, 1408), "true_footprint": (557, 1423)},
        {"det_center": (1323, 483), "true_footprint": (1401, 576)},
        {"det_center": (802, 1475), "true_footprint": (922, 1514)},
    ]
}
CAM_POSITIONS ={
    5: (-0.1, 0.5),
    6: (-0.1, -0.6),
    7: (1, 0),
    8: (0, -2.8),
    9: (2.2, 2.5)
    }
# CAM_HEIGHTS = {5: 1.5,
#                 6: 1.5,
#                 7: 2.5,
#                 8: 2.5,
#                 9: 2.5}
# PIG_AVG_HEIGHT = 0.55                           # Average height of the center of mass of a pig (in meters)
THALES_SCALE = {                                  # Scale to undo bias of each cam (apply Thales theorem)
    5: (0.94, 0.94),                
    6: (0.83, 0.95),
    7: (0.93, 0.93),
    8: (0.93, 0.93),
    9: (0.95, 0.9)
    }

               

# Select detection model and tracking method
DETECTION_METHOD = "YOLO"                       # Options: "YOLO", "ColorSegmentation"
TRACKING_METHOD = "GREEDY"                      # Options: "GREEDY", ("DEEP_SORT")

# Detection settings
YOLO_CONF_THRESHOLD = 0.65                      # Minimum confidence score for detection
YOLO_IMGSZ = 832                                # Image size expected by YOLO model 
LOWER_COLOR_RANGE = [0, 50, 50]                 # Lower bound for color detection in HSV space
UPPER_COLOR_RANGE = [30, 255, 255]              # Upper bound for color detection in HSV space
CONTOUR_AREA_THRESHOLD = 1000                   # Minimum area of contour to be considered a detection

# Tracking settings
MAX_TRACK_AGE = 10                              # Max frames a lost object remains tracked
IOU_THRESHOLD = 0.6                             # Intersection over Union threshold for tracking
MAX_DISTANCE = 50                               # Maximum distance for matching detections to existing tracks   
ALPHA = 0.5                                     # Weight for IoU in cost function, weight of distance is (1 - alpha)
MAX_PIG_MVMT_BETWEEN_TWO_FRAMES = 0.1          # max distance between two points in a batch to be considered from same paths and not outliers

# Video processing settings
FRAME_SKIP = 2                                  # Number of frames to skip for processing

# Output settings
OUTPUT_VIDEO_WIDTH = 1600                       # Width of output video
OUTPUT_VIDEO_HEIGHT = 900                       # Height of output video
OUTPUT_VIDEO_FPS = 20                           # Frames per second for output video

# Mapping settings
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

# Clustering settings
CLUSTER_EPSILON = 0.03                          # Epsilon for DBSCAN clustering -> size of the neighborhood around a point to be considered a cluster
CLUSTER_MIN_SAMPLES = 1                         # Minimum samples for DBSCAN clustering, i.e. minimum number of cameras detecting the same pig for it to be globally tracked
MAX_GLOBAL_AGE = 10                             # Maximum age of a global track (cluster) before it is removed
MAX_CLUSTER_DISTANCE = 0.1                      # Maximum distance between two detections to be considered the same object

# Batch analysis parameters
FRECHET_THRESHOLD = 1                            #TODO
BATCH_SIZE = 100                                 #TODO


"""
OLD MAPPINGS (For distorted images)

MAPPINGS = {    # NOTE: changed scale to 10 strides = 1 in x axis 
    '5' : {
    'image_points': np.array(
        [[1558, 292], [1302, 254], [954, 222], [591, 214], [1108, 1654], [880, 1582], [587, 1457], [97, 1184], [212, 1921], [106, 271]], dtype=np.float32),
    'world_points': np.array(
        [[0, 0], [0.2, 0], [0.5, 0], [0.9, 0], [0.1, 1], [0.3, 1], [0.6, 1], [1.4, 1], [1.3, 2], [1.9, 0]], dtype=np.float32)
                 },
    '6' : {
        'image_points': np.array(
            [[1124, 121], [1502, 146], [1927, 235], [2235, 321], [2444, 414], [1115, 1547], [1606, 1506], [1917, 1464], [2512, 1321], [2283, 1395]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.3, 0], [0.7, 0], [1.1, 0], [1.5, 0], [0, -1], [0.4, -1], [0.7, -1], [1.7, -1], [1.2, -1]], dtype=np.float32)
                    },
    '7' : {
        'image_points': np.array(
            [[527, 891], [873, 895], [1514, 922], [1957, 945], [795, 99], [1227, 70], [1595, 107], [1940, 159], [697, 1644], [1593, 1713]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.5, 0], [1.3, 0], [1.9, 0], [0.3, 1], [0.9, 1], [1.4, 1], [1.9, 1], [0.2, -1], [1.5, -1]], dtype=np.float32)
                    },
    '8' : {
        'image_points': np.array(
            [[460, 1396], [493, 839], [909, 1466], [945, 1020], [1026, 439], [206, 1144], [241, 732], [67, 867], [125, 531], [560, 435]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [1.1, 0], [0.2, -1], [0.9, -1], [1.8, -1], [0.3, 1], [1.3, 1], [0.9, 2], [1.9, 2], [1.9, 0]], dtype=np.float32)
                    },
    '9' : {     # NOTE: camera 9 is actually camera 17 in the farm, we just use 9 for convenience
        'image_points': np.array(
            [[628, 436], [389, 683], [390, 952], [623, 758] ,[636, 1140], [1020, 353], [1020, 826], [1043, 1307], [1691, 493], [1709, 1106]], dtype=np.float32),
        'world_points': np.array(
            [[0, 0], [0.4, -1], [1.1, -1], [0.7, 0], [1.5, 0], [0.1, 1], [0.9, 1], [1.7, 1], [0.5, 2], [1.3, 2]], dtype=np.float32)
                    },
    }
"""