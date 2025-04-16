from src.modules.MultiTracker import MultiTracker
from src.modules.Tracker import Tracker
from src.modules.Mapper import Mapper
from src import config
import pytest

dummy_detection_cam8 = [{'bbox_xyxy': (465, 873, 505, 913), 'center':(485, 893), 'conf': 0.8, 'cls': 0}, {'bbox_xyxy': (1401, 860, 1481, 940), 'center':(1441, 900), 'conf': 0.7, 'cls': 0}]  # (1,0) and outlier
dummy_detection_cam9 = [{'bbox_xyxy': (592, 870, 652, 930), 'center':(622, 900), 'conf': 0.78, 'cls': 0}]  # (1,0)  
dummy_detection_cam6 = [{'bbox_xyxy': (2106, 236, 2226, 356), 'center':(2166, 296), 'conf': 0.78, 'cls': 0}]  # (1,0) 

def test_initialize_multi_tracker():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 4

def test_first_global_track():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4
   
    multi_tracker.globally_match_tracks()
    
    assert len(multi_tracker.globally_tracked) == 1

def test_track_multiple_frames():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4

    multi_tracker.globally_match_tracks()
    assert len(multi_tracker.globally_tracked) == 1
    assert len(multi_tracker.global_detections) == 0
    
    # Simulate tracking in the next frame with the same detections
    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track(dummy_detection_cam9, 9)  # Camera 9
    multi_tracker.track(dummy_detection_cam6, 6)  # Camera 6
    assert len(multi_tracker.global_detections) == 4

    multi_tracker.globally_match_tracks()
    assert len(multi_tracker.globally_tracked) == 1    
    assert len(multi_tracker.global_detections) == 0

def test_no_detections():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    # Simulate no detections in the first frame
    multi_tracker.track([], 8)  # Camera 8
    multi_tracker.track([], 9)  # Camera 9
    multi_tracker.track([], 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 0

    with pytest.raises(ValueError, match="No global detections to match."):
        multi_tracker.globally_match_tracks()

def test_no_clusters():
    multi_tracker = MultiTracker(num_cameras=5, first_camera=5, mapper=Mapper(config.MAPPINGS), cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=5, print_tracked=True)

    # Simulate only 2 detections from cam 8 (no cluster)
    multi_tracker.track(dummy_detection_cam8, 8)  # Camera 8
    multi_tracker.track([], 9)  # Camera 9
    multi_tracker.track([], 6)  # Camera 6

    assert len(multi_tracker.global_detections) == 2

    with pytest.raises(ValueError, match="No clusters found."):
        multi_tracker.globally_match_tracks()
