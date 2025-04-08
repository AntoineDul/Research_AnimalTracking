from src import config
from .detectors.YoloDetector import YoloDetector
from .detectors.ColorSegmentationDetector import ColorSegmentationDetector

class Detector:
    def __init__(self):
        self.detector = self.load_detector()
        self.yolo_conf_threshold = config.YOLO_CONF_THRESHOLD
        self.contour_area_threshold = config.CONTOUR_AREA_THRESHOLD

    def load_detector(self, increment = [0, 0]):
        if config.DETECTION_METHOD == "YOLO":
            return YoloDetector(config.YOLO_MODEL_PATH, config.YOLO_CONF_THRESHOLD + increment[0], config.YOLO_IMGSZ)
        elif config.DETECTION_METHOD == "ColorSegmentation":
            return ColorSegmentationDetector(config.LOWER_COLOR_RANGE, config.UPPER_COLOR_RANGE, config.CONTOUR_AREA_THRESHOLD + increment[1])
        else:
            raise ValueError(f"Unknown model: {config.DETECTION_METHOD}")

    def detect(self, frame):
        return self.detector.detect(frame)
    
    def get_detection_method(self):
        return config.DETECTION_METHOD

    def get_confidence_threshold(self):
        if config.DETECTION_METHOD == "YOLO":
            return self.yolo_conf_threshold
        elif config.DETECTION_METHOD == "ColorSegmentation":
            return self.contour_area_threshold
        
    def increase_threshold(self):
        self.yolo_conf_threshold = min(self.yolo_conf_threshold + 0.05, 1.0)
        self.contour_area_threshold = min(self.contour_area_threshold + 100, 10000)
        self.detector = self.load_detector([self.yolo_conf_threshold, self.contour_area_threshold])

    def decrease_threshold(self):
        self.yolo_conf_threshold = min(self.yolo_conf_threshold - 0.05, 0.01)
        self.contour_area_threshold = min(self.contour_area_threshold - 100, 100)
        self.detector = self.load_detector([self.yolo_conf_threshold, self.contour_area_threshold])

    def reset_threshold(self):
        self.yolo_conf_threshold = config.YOLO_CONF_THRESHOLD
        self.contour_area_threshold = config.CONTOUR_AREA_THRESHOLD
        self.detector = self.load_detector()
