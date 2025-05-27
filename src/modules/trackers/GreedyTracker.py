from .BaseTracker import BaseTracker
from scipy.spatial import distance
import numpy as np


class GreedyTracker(BaseTracker):
    
    def __init__(self, iou_threshold=0.5, max_age=10, max_distance=50, alpha=0.5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_distance = max_distance
        self.alpha = alpha                   # Weight for IoU in cost function
        self.tracked = []
        self.next_id = 1
    
    @staticmethod
    def compute_iou(boxA, boxB):
        """Calculate Intersection over Union (IoU)"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def compute_track_detection_pairs(self, detections):
        """Compute track-detection pairs based on distance and IoU"""

        # intialize cost matrix combining distance between centers of the bboxes and IOU
        cost_matrix = np.zeros((len(self.tracked), len(detections)))

        # Iterate over all tracks and detections to fill the cost matrix
        for i, track in enumerate(self.tracked):
            for j, detection in enumerate(detections):

                center_distance = distance.euclidean(track['center'], detection['center'])
                iou = self.compute_iou(track['bbox_xyxy'], detection['bbox_xyxy'])

                # Check thresholds for distance and IoU
                if center_distance < self.max_distance and iou < self.iou_threshold:

                    # Normalize distance between [0,1] by dividing by max_distance
                    norm_distance = min(center_distance / self.max_distance, 1.0)

                    # Higher IoU is better so use (1 - IoU) as a cost
                    norm_iou = 1 - iou

                    # Combine with weighted sum
                    cost_matrix[i, j] = self.alpha * norm_iou + (1 - self.alpha) * norm_distance
        
        # Sort all track-detection pairs by distance
        track_detection_pairs = []
        for i in range(len(self.tracked)):
            for j in range(len(detections)):
                if cost_matrix[i, j] >= 0 and cost_matrix[i, j] <= 1.0:
                    track_detection_pairs.append((i, j, cost_matrix[i, j]))
        
        # Sort by distance 
        track_detection_pairs.sort(key=lambda x: x[2])

        return track_detection_pairs

    def track(self, detections, print_tracked=False):

        if not self.tracked:
            # first frame, initialize tracks for all detections
            for detection in detections:
                self.tracked.append({
                    "id": self.next_id,
                    "bbox_xyxy": detection["bbox_xyxy"],
                    "center": detection["center"],
                    "conf": detection["conf"],
                    "cls": detection["cls"],
                    "age": 0
                })
                self.next_id += 1
            return self.tracked
        
        # if no detections age all tracks and remove old ones
        if len(detections) == 0:    
            for track in self.tracked:
                track["age"] += 1
            self.tracked = [track for track in self.tracked if track["age"] <= self.max_age]
            return self.tracked
        
        track_detection_pairs = self.compute_track_detection_pairs(detections)
        
        # Track assignments
        assigned_tracks = set()
        assigned_detections = set()

        # Assign detections to tracks
        for track_idx, detection_idx, _ in track_detection_pairs:
            if track_idx not in assigned_tracks and detection_idx not in assigned_detections:
                assigned_tracks.add(track_idx)
                assigned_detections.add(detection_idx)
                
                # Update track with new detection
                self.tracked[track_idx]["bbox_xyxy"] = detections[detection_idx]["bbox_xyxy"]
                self.tracked[track_idx]["center"] = detections[detection_idx]["center"]
                self.tracked[track_idx]["cls"] = detections[detection_idx]["cls"]
                self.tracked[track_idx]["conf"] = detections[detection_idx]["conf"]
                self.tracked[track_idx]["age"] = 0  # Reset age for matched tracks
                
        # Handle unmatched detections (new pigs)
        for j in range(len(detections)):
            if j not in assigned_detections:
                self.tracked.append({
                    "id": self.next_id,
                    "bbox_xyxy": detections[j]["bbox_xyxy"],
                    "center": detections[j]["center"],
                    "cls": detections[j]["cls"],
                    "conf": detections[j]["conf"],
                    "age": 0
                })
                self.next_id += 1
        
        # Increase age for unmatched tracks
        for i in range(len(self.tracked)):
            if i not in assigned_tracks:
                self.tracked[i]["age"] += 1
        
        # Remove old tracks
        self.tracked = [track for track in self.tracked if track["age"] <= self.max_age]
        
        if print_tracked:
            print(f"Tracked pigs: {len(self.tracked)}")
            for track in self.tracked:
                print(f"Track ID: {track['id']}, Age: {track['age']}, BBox: {track['bbox_xyxy']}, Center: {track['center']}, Conf: {track['conf']}")
        
        return self.tracked
    
    def get_tracks(self):
        return self.tracked

    def reinitialize_id_count(self):
        self.next_id = 1
        return