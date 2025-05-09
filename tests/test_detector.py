import pytest
from src.modules.Detector import Detector 

def test_detector_initialization_and_outputs():
    detector = Detector()
    assert detector is not None, "Detector should be initialized successfully."

    detections = detector.detect("outputs/undistorted_cameras/camera_17_k1_-0.5_k2_0.1.png")
    assert detections is not None, "Detector should return detections."

    assert isinstance(detections, list), "Detections should be a list."
    assert len(detections) > 0, "Detections list should not be empty."
    
    
    for detection in detections:
        print("------------")
        print(detection)

        # Check if each detection has the required keys
        assert isinstance(detection, dict), "Each detection should be a dictionary."
        assert "bbox_xyxy" in detection, "Detection should contain 'bbox_xyxy'."
        assert "center" in detection, "Detection should contain 'center'."
        assert "conf" in detection, "Detection should contain 'conf'."
        assert "cls" in detection, "Detection should contain 'cls'."