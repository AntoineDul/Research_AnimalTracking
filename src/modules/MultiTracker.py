from sklearn.cluster import DBSCAN
import numpy as np
import json
from scipy.spatial import distance
from src.modules.Tracker import Tracker

class MultiTracker:
    def __init__(self, num_cameras=4, first_camera=5, mapper=None, cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=1, print_tracked=False):
        self.trackers = [Tracker() for _ in range(num_cameras)]
        self.mapper = mapper    
        self.num_cameras = num_cameras
        self.first_camera = first_camera

        # cluster parameters
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples

        # tracker parameters
        self.max_age = max_age                                   # Maximum age of a track before it is removed -> might remove later
        self.max_cluster_distance = max_cluster_distance         # Maximum distance between two detections to be considered the same object
        self.global_detections = []                              # List of all detections from all cameras in real world coordinates
        self.globally_tracked = []                               # List of all tracked pigs in real world coordinates
        self.global_id_mapping = {}
        self.print_tracked = print_tracked

        self.tracking_history = []
        self.frame = 0
    def convert_to_builtin_type(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_builtin_type(x) for x in obj)
        elif isinstance(obj, list):
            return [self.convert_to_builtin_type(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_to_builtin_type(v) for k, v in obj.items()}
        else:
            return obj
    
    def save_tracking_history(self, file_path):
        """Save the tracking history to a JSON file."""

        print(f"Tracking history : {self.tracking_history}")

        clean_history = self.convert_to_builtin_type(self.tracking_history)
        with open(file_path, 'w') as f:
            json.dump(clean_history, f, indent=4)
        print(f"Tracking history saved to {file_path}")
    
    def track(self, detections, cam_id):
        """Track detections from a SINGLE camera and update global detections."""
        
        # Mono camera tracking
        if cam_id is None:
            tracks = self.trackers[0].track(detections)  # Default to one camera tracker
        
        # Multi camera tracking
        else:
            tracks = self.trackers[cam_id % self.first_camera].track(detections)    # Use right tracker for camera

            if self.mapper is not None:
                for track in tracks:
                    world_coordinates = self.mapper.image_to_world_coordinates(cam_id, np.array([[[track['center'][0], track['center'][1]]]], dtype=np.float32))
                    self.global_detections.append({
                        'cam_id': cam_id,
                        'local_id': track['id'],
                        'world_coordinates': world_coordinates,
                      })             
            # print(f"GLOBAL DETECTIONS: {self.global_detections}")
        return tracks     
           
    def globally_match_tracks(self):
        """Once all detections from all cameras have been tracked, match them to global tracks using clustering."""

        if len(self.globally_tracked) != 0:
            self.tracking_history.append({'global tracks': self.globally_tracked.copy(), 'frame' : self.frame})   # 
            self.frame += 1     # NOTE: This is the number of frames processed, need to do times 2 to get real number since we skip every 2nd frame

        all_coords = np.array([d['world_coordinates'] for d in self.global_detections])

        if len(all_coords) == 0:
            raise ValueError("No global detections to match.")
        
        # Group all gloal detections using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering
        clusters = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples).fit(all_coords)

        # Compute cluster center coordinates and labels pairs
        cluster_center_label_pairs = self.get_center_label_pairs(clusters, all_coords)      # [((x, y), cluster_id), ...]

        # print(f"Cluster centers: {cluster_center_label_pairs}")

        if len(cluster_center_label_pairs) == 0:
            raise ValueError("No clusters found.")
        
        self.match_clusters_with_previous_tracks(cluster_center_label_pairs)  # Update global detections with cluster centers
        return self.globally_tracked

# Helper functions
    def match_clusters_with_previous_tracks(self, cluster_center_label_pairs): 
        # Update global detections with cluster centers
        if not self.globally_tracked:
            # first frame, initialize tracks for all detections
            for center, label in cluster_center_label_pairs:
                
                # Skip noise points
                if label == -1:
                    continue

                # detection['global_id'] = int(label)
                self.globally_tracked.append({
                    "id": label,
                    "center": center,
                    "age": 0
                })
            self.global_detections = []  # Clear global detections after processing
            return self.globally_tracked
        
        # if no detections age all tracks and remove old ones
        if len(cluster_center_label_pairs) == 0:    
            for track in self.globally_tracked:
                track["age"] += 1
            self.globally_tracked = [track for track in self.globally_tracked if track["age"] <= self.max_age]
            self.global_detections = []  # Clear global detections after processing
            return self.globally_tracked
        
        track_cluster_pairs = self.compute_track_cluster_pairs(cluster_center_label_pairs)

        print("**************************")
        print(f"track cluster pairs : {track_cluster_pairs}")

        # Track assignments
        assigned_tracks = set()
        assigned_clusters = set()

        # Assign detections to tracks
        for track_idx, cluster_label_idx, _ in track_cluster_pairs:
            cluster_label = cluster_center_label_pairs[cluster_label_idx][1]
            cluster_center = cluster_center_label_pairs[cluster_label_idx][0]
            track_label = self.globally_tracked[track_idx]["id"]
            print(f"cluster label : {cluster_label}\ntrack label : {track_label}")
            if track_label not in assigned_tracks and cluster_label not in assigned_clusters:
                assigned_tracks.add(track_label)
                assigned_clusters.add(cluster_label)
                
                # Update track with new detection
                self.globally_tracked[track_idx]["center"] = cluster_center
                self.globally_tracked[track_idx]["age"] = 0  # Reset age for matched tracks
                
        # Handle unmatched detections (new pigs)
        for j in range(len(cluster_center_label_pairs)):
            if cluster_center_label_pairs[j][1] not in assigned_clusters:
                self.globally_tracked.append({
                    "id": cluster_center_label_pairs[j][1],
                    "center": cluster_center_label_pairs[j][0],
                    "age": 0
                })
        
        # Increase age for unmatched tracks
        for i in range(len(self.globally_tracked)):
            if self.globally_tracked[i]['id'] not in assigned_tracks:
                self.globally_tracked[i]["age"] += 1
        
        # Remove old tracks
        self.globally_tracked = [track for track in self.globally_tracked if track["age"] <= self.max_age]
        
        if self.print_tracked:
            print(f"Tracked pigs: {len(self.globally_tracked)}")
            for track in self.globally_tracked:
                print(f"Track ID: {track['id']}, Age: {track['age']}, Center: {track['center']}")

        self.global_detections = []  # Clear global detections after processing
        return self.globally_tracked

    def compute_track_cluster_pairs(self, clusters_label_pairs):
        # NOTE: only based on distance between cluster centers and track centers, might not be very accurate

        # intialize cost matrix 
        cost_matrix = np.zeros((len(self.globally_tracked), len(clusters_label_pairs)))

        print(f"compute track cluster pairs\n======================================\nglobally tracked: {self.globally_tracked} \n clusters label pair : {clusters_label_pairs} \n ")

        # Iterate over all tracks and detections to fill the cost matrix
        for i, track in enumerate(self.globally_tracked):
            for j, (center, label) in enumerate(clusters_label_pairs):
                print("TRACK: ", track)
                print(f"Track center {i}: {track['center']}, Cluster center {j}: {center}, frame: {self.frame}")
                print("==========================================")

                cluster_distance = distance.euclidean(track['center'], center)

                # Check thresholds for distance and IoU
                if cluster_distance < self.max_cluster_distance:

                    # Normalize distance between [0,1] by dividing by max_distance
                    norm_distance = min(cluster_distance / self.max_cluster_distance, 1.0)

                    # Combine with weighted sum
                    cost_matrix[i, j] = norm_distance
        
        # Sort all track-detection pairs by distance
        track_detection_pairs = []
        for i in range(len(self.globally_tracked)):     # Index of tracks
            for j in range(len(clusters_label_pairs)):  # Index of pair 
                if cost_matrix[i, j] >= 0 and cost_matrix[i, j] <= 1.0:
                    track_detection_pairs.append((i, j, cost_matrix[i, j]))
        
        # Sort by distance 
        track_detection_pairs.sort(key=lambda x: x[2])

        return track_detection_pairs

    def get_center_label_pairs(self, clusters, all_coords):
        labels = clusters.labels_   # array of cluster labels for each point (-1 for noise) -> eg: array([0, 0, 0, 1, 1, 2, -1, 1, 2, 0])
        unique_labels = list(set(label for label in labels if label != -1))
        cluster_center_label_pairs = []  # list of tuples (cluster center id, cluster id) for all cluster centers

        # Compute the center of each cluster
        for label in unique_labels:
            points_in_cluster = all_coords[labels == label]
            center = points_in_cluster.mean(axis=0)
            cluster_center_label_pairs.append(((center[0], center[1]), label))      # [((x, y), cluster_id), ...]

        return cluster_center_label_pairs
