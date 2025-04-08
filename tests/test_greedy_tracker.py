import pytest
import numpy as np
from scipy.spatial import distance
from src.modules.trackers.GreedyTracker import GreedyTracker 

# Sample detection data
def get_sample_detections():
    return [
        {"bbox_xyxy": [10, 10, 20, 20], "center": [15, 15], "conf": 0.9, "cls": 0},
        {"bbox_xyxy": [30, 30, 40, 40], "center": [35, 35], "conf": 0.8, "cls": 0},
    ]

# GreedyTracker(iou_threshold=0.5, max_age=10, max_distance=50, alpha=0.5)

# Test 1: Initialize tracker and check initial state
def test_initialization():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    tracks = tracker.track(detections)
    assert len(tracks) == 2  # Two detections should be tracked
    assert tracks[0]["id"] == 1
    assert tracks[1]["id"] == 2

# Test 2: Tracker should not assign new detections if no change in the second frame
def test_no_change_in_second_frame():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    tracks = tracker.track(detections)  # First frame
    tracks = tracker.track(detections)  # Second frame (same detections)
    assert len(tracks) == 2             # No new tracks, just updated positions
    assert tracks[0]["id"] == 1
    assert tracks[1]["id"] == 2

# Test 3: Tracker should update the track if detections move (i.e., center changes)
def test_track_update_on_center_move():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    tracks = tracker.track(detections)  # First frame
    
    # Simulate some movement in detections
    detections[0]["bbox_xyxy"] = [12, 13, 24, 19] 
    detections[1]["bbox_xyxy"] = [28, 30, 33, 36]  
    detections[0]["center"] = [18, 16]
    detections[1]["center"] = [31, 33]
    
    tracks = tracker.track(detections)  # Second frame (updated detections)
    
    # Check if track centers are updated
    assert len(tracks) == 2  # Still two tracks
    assert tracks[0]["id"] == 1
    assert tracks[1]["id"] == 2
    assert tracks[0]["center"] == [18, 16]
    assert tracks[1]["center"] == [31, 33]

# Test 4: Handle unmatched detections
def test_handle_unmatched_detections():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    tracks = tracker.track(detections)  # First frame
    
    # Add a new detection
    new_detection = {"bbox_xyxy": [90, 100, 110, 120], "center": [100, 110], "conf": 0.7, "cls": 0}
    tracks = tracker.track(detections + [new_detection])  # Second frame with a new detection
    
    assert len(tracks) == 3  # One new track should be added
    assert tracks[-1]["id"] == 3  # The new detection should get a new ID

# Test 5: Test IOU calculation for two overlapping boxes
def test_iou_calculation():
    boxA = [10, 10, 20, 20]
    boxB = [15, 15, 25, 25]
    tracker = GreedyTracker()
    iou = tracker.compute_iou(boxA, boxB)
    assert iou > 0.0  # Assert that the IoU is correctly calculated (expecting overlap)

# Test 6: Test cost matrix calculation with distance and IOU
def test_cost_matrix_computation():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    
    # Track initial detections
    tracker.track(detections)
    
    # Create a new set of detections for the next frame
    new_detections = [
        {"bbox_xyxy": [11, 21, 31, 23], "center": [21, 22], "conf": 0.9, "cls": 1},
        {"bbox_xyxy": [51, 61, 71, 81], "center": [61, 71], "conf": 0.8, "cls": 1},
    ]
    
    # Compute track-detection pairs
    track_detection_pairs = tracker.compute_track_detection_pairs(new_detections)
    
    assert len(track_detection_pairs) > 0  # Should find at least one pair

# Test 7: Handle tracks aging out
def test_tracks_aging_out():
    tracker = GreedyTracker()
    detections = get_sample_detections()
    tracks = tracker.track(detections)  # First frame
    tracks = tracker.track(detections)  # Second frame
    tracks = tracker.track(detections)  # Third frame
    
    # After 3 frames, the tracks should still be there
    assert len(tracks) == 2
    assert tracks[0]["age"] == 0
    assert tracks[1]["age"] == 0
    
    for i in range(15):
        # Simulate aging out by not providing detections
        tracks = tracker.track([])
    
    assert len(tracks) == 0  # Tracks should be gone since max_age is 10 by default

