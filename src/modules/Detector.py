from src import config
from .detectors.YoloDetector import YoloDetector

class Detector:
    def __init__(self):
        self.detector = self.load_detector()

    def load_detector(self):
        if config.DETECTION_METHOD == "YOLO":
            return YoloDetector(config.YOLO_MODEL_PATH, config.YOLO_CONF_THRESHOLD, config.YOLO_IMGSZ)
        else:
            raise ValueError(f"Unknown model: {config.DETECTION_METHOD}")

    def detect(self, frame):
        return self.detector.detect(frame)
    