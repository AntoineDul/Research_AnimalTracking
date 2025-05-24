from sklearn.cluster import DBSCAN
import numpy as np
import math
# from frechetdist import frdist
from shapely import LineString, frechet_distance
import json
from scipy.spatial import distance
from src.modules.Tracker import Tracker

class MultiTracker:

    def __init__(self, num_cameras=5, first_camera=5, mapper=None, cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=1, frechet_threshold=3, batch_size=10, overlapped_cams=[5, 6, 7],print_tracked=False):
        self.trackers = [Tracker() for _ in range(num_cameras)]
        self.mapper = mapper    
        self.num_cameras = num_cameras
        self.first_camera = first_camera
        self.overlapped_cams = overlapped_cams

        # cluster parameters
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.max_cluster_distance = max_cluster_distance         # Maximum distance between two detections to be considered the same object

        # tracker parameters
        self.max_age = max_age                                   # Maximum age of a track before it is removed -> might remove later
        self.global_detections = []                              # List of all detections from all cameras in real world coordinates
        self.globally_tracked = []                               # List of all tracked pigs in real world coordinates
        self.global_id_mapping = {}
        self.global_id = 0
        self.print_tracked = print_tracked

       # Batch analysis parameters
        self.frechet_threshold = frechet_threshold               # Max distance between 2 paths to be analyzed
        self.batch_size = batch_size                             # Number of frames to analyze in one "batch"
        self.global_detections_by_cam = [self.num_cameras * []]  # List of lists of detections over a batch by each camera
        self.batch_path_by_cam = [self.num_cameras * []]         # List of paths by id for each camera -> [[[cam1_path1], [cam1_path2]], [[cam2_path1], [cam2_path2]], ...]
        self.global_batches_tracks = None                        # Overall "fusion" of each batch paths appended here 

        # for logs 
        self.pov_tracking_by_cam = {}
        self.global_tracking_by_cam = {}
        self.tracking_history = []
        self.frame = 0
    
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
                max_id = 0
                for track in tracks:
                    if track['id'] > max_id :
                        max_id = track['id']
                    world_coordinates = self.mapper.image_to_world_coordinates(cam_id, np.array([[[track['center'][0], track['center'][1]]]], dtype=np.float32))
                    self.global_detections.append({
                        'cam_id': cam_id,
                        'local_id': track['id'],
                        'world_coordinates': world_coordinates,
                      })
                    
                #     self.global_detections_by_cam[cam_id % self.first_camera].append({
                #         'cam_id': cam_id,
                #         'local_id': track['id'],
                #         'world_coordinates': world_coordinates,
                #     })     

                # # TODO: TEST THIS
                # for cam in range(self.num_cameras):
                #     for id in range(max_id):
                #         id_path = []
                #         for detection in self.global_detections_by_cam[cam]:
                #             if detection['local_id'] == id:
                #                 id_path.append(detection['world_coordinates'])
                #         self.batch_path_by_cam[cam].append(id_path)
                        
            # print(f"GLOBAL DETECTIONS: {self.global_detections}")
        return tracks     
           
    def globally_match_tracks(self):
        """Once all detections from all cameras have been tracked, match them to global tracks using clustering."""

        if len(self.globally_tracked) != 0:
            history = {'global tracks': self.globally_tracked.copy(), 'frame' : self.frame}
            print(f"\nFRAME {self.frame}: {history}\n")
            self.tracking_history.append(history)   # 
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

        # Track assignments
        assigned_tracks = set()
        assigned_clusters = set()

        # Assign detections to tracks
        for track_idx, cluster_label_idx, _ in track_cluster_pairs:
            cluster_label = cluster_center_label_pairs[cluster_label_idx][1]
            cluster_center = cluster_center_label_pairs[cluster_label_idx][0]
            track_label = self.globally_tracked[track_idx]["id"]
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

        # Iterate over all tracks and detections to fill the cost matrix
        for i, track in enumerate(self.globally_tracked):
            for j, (center, label) in enumerate(clusters_label_pairs):

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

    def batch_match(self, paths_list1, paths_list2, cam_id1, cam_id2):  # paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        """Match similar paths from 2 cameras"""

        # NOTE: possible issue: path taken as inputs are tracks from each camera, may contain mistakes (mislabeled pigs leading to wrong track, etc...)
        # Solution:
        # TODO: if 2 "different" paths from a pov dont have frames in common and match to a single path from another pov, could be the same pig taht was wrongly 
        # re id in one pov -> handle this. 
        
        # TODO: store coords of path of each cams to plot a bunch of graph : 2 per cams (cam coords and global coords, 2 self vars created in init) and then plot of global movements 
        # TODO: think about how to overlap batches for more accuracy

        # TODO: extenslively test batch_match
        # TODO: add biases to cam views
        
        global_paths = []

        if self.batch_size == None:
            raise ValueError("Missing batch_size parameter.")

        # Handle none values for paths -> sometimes there are no pigs in pov
        if paths_list1 == None and paths_list2 is not None:
            return paths_list2
        elif paths_list2 == None and paths_list1 is not None:
            return paths_list1
        elif paths_list1 == None and paths_list2 == None:
            return None
        
        cost_matrix = np.zeros((len(paths_list1), len(paths_list2)))

        for i, path1 in enumerate(paths_list1):
            for j, path2 in enumerate(paths_list2):

                if len(path1) < 2 or len(path2) <2:
                                    continue        # If single point, no path to compare
                
                # Separate points coordinates from frame numbers to compute frechet dist
                points_path1 = [t[0] for t in path1]
                points_path2 = [t[0] for t in path2]

                # print(f"points path 1 : {points_path1}\n points path 2 : {points_path2}\n")

                paths_dist = frechet_distance(LineString(points_path1), LineString(points_path2))

                # print(f"FRECHET DISTANCE BETWEEN \n {path1} \n AND \n{path2}\n: {paths_dist}\n\n")

                cost_matrix[i, j] = paths_dist
        
        paths_pairs = []
        for i in range(len(paths_list1)):
            for j in range(len(paths_list2)):
                if cost_matrix[i, j] <= self.frechet_threshold:
                    paths_pairs.append((paths_list1[i], paths_list2[j], cost_matrix[i, j]))
        
        # Sort by Frechet distance
        paths_pairs.sort(key=lambda x:x[2])

        # If path appears only once -> keep (might be a pig visible in only one pov) 
        # -> should this apply to all cams or only ones that have zone with no overlaps
        # If path has matches -> keep only the best one 

        # Track assignments
        assigned_paths1 = set()
        assigned_paths2 = set()


        for path1, path2, _ in paths_pairs:
            path1 = tuple(path1)     # need to convert to tuples because lists are not hashable
            path2 = tuple(path2)
            if path1 not in assigned_paths1 and path2 not in assigned_paths2:  
                assigned_paths1.add(path1)
                assigned_paths2.add(path2)

                merged_path = self.merge_paths(path1, path2, self.batch_size)
                global_paths.append(merged_path)

        # Add path that could potentially only be seen by one cam
        if cam_id1 not in self.overlapped_cams:
            for path1 in paths_list1:
                path1 = tuple(path1)
                if path1 not in assigned_paths1 and path1 not in global_paths:
                    global_paths.append(path1)
                    assigned_paths1.add(path1)
         
        if cam_id2 not in self.overlapped_cams:
            for path2 in paths_list2:
                path2 = tuple(path2)
                if path2 not in assigned_paths2 and path2 not in global_paths:
                    global_paths.append(path2)
                    assigned_paths2.add(path2)

        return global_paths
         
    @staticmethod
    def merge_paths(path1, path2, batch_size):      #  path = [((x1, y1), f1), ((x2, y2), f2), ...] NOTE: should be sorted by frame numbers f1<f2<f3...
        frames1 = []            # List of frames present in path 1
        frames2 = []            # List of frames present in path 2

        avg_frames = []         # List of frame for which we need to take avg of coords
        cam1_only_frames = []   # List of frames only detected by cam1
        cam2_only_frames = []   # List of frames only detected by cam2

        final_path = []         # final result

        for point in path1:
            frames1.append(point[1])

        for point in path2:
            frames2.append(point[1])

        for i in range(batch_size):
            if i in frames1 and i in frames2:
                avg_frames.append(i)
            elif i in frames1 and i not in frames2:
                cam1_only_frames.append(i)
            elif i not in frames1 and i in frames2:
                cam2_only_frames.append(i)
        
        # print(f"path 1 : {path1}\n path 2 : {path2}\navg_frames : {avg_frames}\ncam1 only frames : {cam1_only_frames}\n cam2 only frames :{cam2_only_frames}\n")
        
        idx_path1 = 0
        idx_path2 = 0
        
        for i in range(batch_size):
            
            # Once all the data from a list has been used, it will skip these assignments
            if idx_path1 < len(path1): 
                coords1 = path1[idx_path1][0]
                frame1 = path1[idx_path1][1]

            if idx_path2 < len(path2):
                coords2 = path2[idx_path2][0]
                frame2 = path2[idx_path2][1]

            if i in cam1_only_frames:
                assert frame1 == i
                final_path.append(((path1[idx_path1][0]), i))
                idx_path1 += 1

            elif i in cam2_only_frames:
                assert frame2 == i
                final_path.append(((path2[idx_path2][0]), i))
                idx_path2 += 1

            elif i in avg_frames:
                assert frame1 == i
                assert frame2 == i
                final_path.append((((((coords1[0] + coords2[0])/2), ((coords1[1] + coords2[1])/2)), i)))
                idx_path1 += 1
                idx_path2 += 1

            else: 
                # NOTE: could defined some behaviour when pig disappears for a frame 
                continue

        return final_path

    def handle_outliers(self, max_mvmt_between_frames, paths_by_cam):
        for cam_id, paths_list in paths_by_cam.items():
            outliers = []
            frames_present_in_paths = []
            for path in paths_list:
                frames_present = []
                i = 0
                while i + 1 < len(path) and path[i + 1]:
                    point1 = path[i]
                    point2 = path[i + 1]
                    # print(f"point 1 (i={i}): {point1}\npoint2 (i+1 = {i+1}) : {point2}")
                    frames_present.append(point1[1])
                    # print(f"distance between points: {self.euclidean_distance(point1[0], point2[0])}")
                    if self.euclidean_distance(point1[0], point2[0]) > max_mvmt_between_frames: 
                        # print(f"outlier detected: {point2}")
                        outliers.append(point2)
                        path.remove(point2)
                    else:
                        frames_present.append(point2[1])
                        i += 1
                frames_present_in_paths.append(set(frames_present))

            outliers.sort(key=lambda outlier: outlier[1])
            # print("-------------------------------- rebuilding")
            # print(f"paths list: {paths_list}\n\n frames present:{frames_present_in_paths}\n")
            # print(f"OUTLIERS (sorted): {len(outliers)} outliers\n {outliers}")
            outliers_placed = []
            for outlier in outliers:
                # print(f"checking outlier: {outlier}") 
                for path_idx, frames_path in enumerate(frames_present_in_paths):
                    frame_number_outlied = outlier[1]
                    coords_outlied = outlier[0]
                    if frame_number_outlied not in frames_path and len(frames_path) > 1:

                        if frame_number_outlied - 1 >= 0 and frame_number_outlied - 1 < len(paths_list[path_idx]):
                            # print(f"comparing {coords_outlied} and {paths_list[path_idx][frame_number_outlied-1][0]}\n euclidean distance = {self.euclidean_distance(coords_outlied, paths_list[path_idx][frame_number_outlied - 1][0]) }")
                            if self.euclidean_distance(coords_outlied, paths_list[path_idx][frame_number_outlied - 1][0]) < max_mvmt_between_frames:
                                paths_list[path_idx].insert(frame_number_outlied, outlier)
                                outliers_placed.append(outlier)
                                break

            for outlier_placed in outliers_placed:
                outliers.remove(outlier_placed)
            # print(f"\noutliers remaining: {len(outliers)} outliers left\n{outliers}")
        return paths_by_cam

    @staticmethod
    def euclidean_distance(p1, p2):
        """Calculate the Euclidean distance between two 2D points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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