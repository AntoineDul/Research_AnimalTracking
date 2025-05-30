from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
from shapely import LineString, frechet_distance
import json
from scipy.spatial import distance
from src.modules.Tracker import Tracker

class MultiTracker:

    def __init__(self, num_cameras=5, first_camera=5, mapper=None, cluster_eps=0.5, cluster_min_samples=2, max_age=5, max_cluster_distance=1, frechet_threshold=3, similarity_threshold=0.05, batch_size=10, overlapped_cams=[5, 6, 7], non_overlap_threshold=[-1.5, 1], print_tracked=False):
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
        self.similarity_threshold = similarity_threshold         # Min frechet distance for 2 paths to be considered distinct
        self.batch_size = batch_size                             # Number of frames to analyze in one "batch"
        self.global_detections_by_cam = [self.num_cameras * []]  # List of lists of detections over a batch by each camera
        self.batch_path_by_cam = [self.num_cameras * []]         # List of paths by id for each camera -> [[[cam1_path1], [cam1_path2]], [[cam2_path1], [cam2_path2]], ...]
        self.global_batches_tracks = None                        # Overall "fusion" of each batch paths appended here 
        self.orphan_paths = []                                   # Paths that have not found a match yet, they will be checked for the remaining iterations
        self.non_overlap_threshold = non_overlap_threshold       # Defining zones of the pen that can only be seen by one camera
        # max_frame_gap = 10                                     # Max time gap between
        self.max_path_length_ratio = 0.3                         # Max lengths ratio acceptable to match paths (avoid matching 2 points with a 20 points path)

        # for logs 
        self.pov_tracking_by_cam = {}
        self.global_tracking_by_cam = {}
        self.tracking_history = []
        self.frame = 0
    
    def save_tracking_history(self, file_path):
        """Save the tracking history to a JSON file."""

        # print(f"Tracking history : {self.global_batches_tracks}")

        clean_history = self.convert_to_builtin_type(self.global_batches_tracks)
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
           
# ============== Cluster implementation (not very efficient) =============
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

# =============== Batch implementation ==============================
    def batch_match(self, paths_list1, paths_list2, cam_id1, cam_id2):  # paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        """
        Match and merge similar paths from two different camera views.
        paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        """



        # TODO: if 2 "different" paths from a pov dont have frames in common and match to a single path from another pov, could be the same pig taht was wrongly 
        # re id in one pov -> handle this. 
        
        # TODO: store coords of path of each cams to plot a bunch of graph : 2 per cams (cam coords and global coords, 2 self vars created in init) and then plot of global movements 
        # TODO: think about how to overlap batches for more accuracy

        # TODO: extenslively test batch_match
        
        # Check parameters
        if self.batch_size == None:
            raise ValueError("Missing batch_size parameter.")

        # Handle none values for paths -> sometimes there are no pigs in pov
        if paths_list1 == None and paths_list2 is not None:
            return paths_list2
        elif paths_list2 == None and paths_list1 is not None:
            return paths_list1
        elif paths_list1 == None and paths_list2 == None:
            return None

        # Declare final paths list
        global_paths = []  

        empty_paths1 = 0
        empty_paths2 = 0

        # Initialize cost matrix
        len_paths_1 = len(paths_list1)
        len_paths_2 = len(paths_list2)
        cost_matrix = np.full((len_paths_1, len_paths_2), np.inf)

        for i, path1 in enumerate(paths_list1):
            for j, path2 in enumerate(paths_list2):

                # If single point, no path to compare
                if len(path1) < 2:
                    empty_paths1 += 1   # keep track of number of empty paths
                    continue

                if len(path2) <2:
                    empty_paths2 += 1   # keep track of number of empty paths
                    continue       
                
                # Separate points coordinates from frame numbers to compute frechet dist
                coords1 = [p[0] for p in path1]
                coords2 = [p[0] for p in path2]

                # Compute Frechet distance between paths
                paths_dist = frechet_distance(LineString(coords1), LineString(coords2))

                # Fill cost matrix
                cost_matrix[i, j] = paths_dist

        # Check if cost matrix is not empty
        if np.all(np.isinf(cost_matrix)):
            if empty_paths1 == len_paths_1 and empty_paths2 == len_paths_2:
                return paths_list1
            elif empty_paths1 == len_paths_1:
                return paths_list2
            elif empty_paths2 == len_paths_2:
                return paths_list1
            else: 
                raise ValueError

        valid_pairs = [(i, j) for i in range(len_paths_1) for j in range(len_paths_2) if not np.isinf(cost_matrix[i, j])]

        if not valid_pairs:
            raise ValueError

        # Build submatrix
        rows = sorted(set(i for i, _ in valid_pairs))
        cols = sorted(set(j for _, j in valid_pairs))
        submatrix = cost_matrix[np.ix_(rows, cols)]

        row_ind_sub, col_ind_sub = linear_sum_assignment(submatrix)

        # Map back to original indices
        row_ind = [rows[i] for i in row_ind_sub]
        col_ind = [cols[j] for j in col_ind_sub]

        # Apply Hungarian algorithm
        # row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # paths_pairs = []
        # for i in range(len(paths_list1)):
        #     for j in range(len(paths_list2)):
        #         if cost_matrix[i, j] <= self.frechet_threshold:
        #             paths_pairs.append((paths_list1[i], paths_list2[j], cost_matrix[i, j]))
        
        # # Sort by Frechet distance
        # paths_pairs.sort(key=lambda x:x[2])

        # If path appears only once -> keep (might be a pig visible in only one pov) 
        # -> should this apply to all cams or only ones that have zone with no overlaps
        # If path has matches -> keep only the best one 

        # Track assignments
        assigned_paths1 = set()
        assigned_paths2 = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.frechet_threshold:
                merged_path = self.merge_paths(paths_list1[i], paths_list2[j], self.batch_size)
                global_paths.append(merged_path)
                assigned_paths1.add(i)
                assigned_paths2.add(j)

        # Function to check if a pig is in a zone covered by only one cam 
        def is_outside_overlap(path):
            avg_y = (path[0][0][1] + path[-1][0][1]) / 2
            return avg_y <= self.non_overlap_threshold[0] or avg_y >= self.non_overlap_threshold[1] 

        # Iterate through the remaining non assigned paths
        for i, path in enumerate(paths_list1):
            if i not in assigned_paths1:
                # Add paths if it is in a zone covered only by that camera
                if cam_id1 not in self.overlapped_cams or is_outside_overlap(path):
                    global_paths.append(path)
                
                # Check with orphans paths (non assigned paths) from previous cameras
                else:
                    merged_path, found = self.check_orphan_paths(path)
                    if found:
                        global_paths.append(merged_path)
                    
                    # If no match, add to orphan list
                    else:
                        self.orphan_paths.append(path)

        # Repeat for second paths list
        for j, path in enumerate(paths_list2):
            if j not in assigned_paths2:
                if cam_id2 not in self.overlapped_cams or is_outside_overlap(path):
                    global_paths.append(path)
                else:
                    merged_path, found = self.check_orphan_paths(path)
                    if found:
                        global_paths.append(merged_path)
                    else:
                        self.orphan_paths.append(path)

        return global_paths

        # for path1, path2, _ in paths_pairs:
        #     path1_id = id(path1)     # need to convert to ids because lists are not hashable
        #     path2_id = id(path2)
        #     if path1_id not in assigned_paths1 and path2_id not in assigned_paths2:  
        #         assigned_paths1.add(path1_id)
        #         assigned_paths2.add(path2_id)

        #         merged_path = self.merge_paths(path1, path2, self.batch_size)
        #         global_paths.append(merged_path)

        # # Add path that could potentially only be seen by one cam
        # if cam_id1 not in self.overlapped_cams:
        #     for path1 in paths_list1:
        #         first_path_point = path1[0][0]
        #         last_path_point = path1[-1][0]
        #         if (((first_path_point[1] + last_path_point[1]) / 2) <= -1.5) or (((first_path_point[1] + last_path_point[1]) / 2) >= 1):
        #             path1_id = id(path1)
        #             if path1_id not in assigned_paths1 and path1_id not in global_paths:
        #                 global_paths.append(path1)
        #                 assigned_paths1.add(path1_id)
        #         else:
        #             merged_path, found = self.check_orphan_paths(path1)
        #             if not found:
        #                 self.orphan_paths.append(path1)
        #                 # print(f"\norphan path : {path1}\ncam id {cam_id1}, avgpoint : {((first_path_point[1] + last_path_point[1]) / 2)}")

        #             elif found:
        #                 global_paths.append(merged_path)
         
        # if cam_id2 not in self.overlapped_cams:
        #     for path2 in paths_list2:
        #         first_path_point = path2[0][0]
        #         last_path_point = path2[-1][0]

        #         if (((first_path_point[1] + last_path_point[1]) / 2) <= -1.5) or (((first_path_point[1] + last_path_point[1]) / 2) >= 1):
        #             path2_id = id(path2)
        #             if path2_id not in assigned_paths2 and path2_id not in global_paths:
        #                 global_paths.append(path2)
        #                 assigned_paths2.add(path2_id)
                        
        #         else:
        #             merged_path, found = self.check_orphan_paths(path2)
        #             if not found:
        #                 self.orphan_paths.append(path2)
        #                 # print(f"\norphan path : {path2}\ncam id {cam_id2}, avgpoint : {((first_path_point[1] + last_path_point[1]) / 2)}")

        #             elif found:
        #                 global_paths.append(merged_path)

        # return global_paths

    def check_orphan_paths(self, path_to_check):
        """Check if some paths that were not matched in previous batches can be match in current batch"""

        # If path is a single point, discard
        if len(path_to_check) < 2:
            return None, False

        # Convert path to check in a LineString object to compute frechet distance
        points_path1 = [pt[0] for pt in path_to_check]
        line1 = LineString(points_path1)

        # Initialize variables
        best_match = None
        best_distance = np.inf
        best_idx = -1

        for j, orphan_path in enumerate(self.orphan_paths):
            # If path is a single point, discard
            if len(orphan_path) < 2:
                continue
            
            # # Frame-based filter
            # start_check = path_to_check[0][1]
            # end_check = path_to_check[-1][1]
            # start_orphan = orphan_path[0][1]
            # end_orphan = orphan_path[-1][1]
            # frame_gap = min(abs(start_check - start_orphan), abs(end_orphan - end_check))
            # if frame_gap > self.max_frame_gap:
            #     continue

            # Length consistency filter
            len_ratio = min(len(path_to_check), len(orphan_path)) / max(len(path_to_check), len(orphan_path))
            if len_ratio < self.max_path_length_ratio:
                continue

            # Convert to LineString object
            points_path2 = [pt[0] for pt in orphan_path]
            line2 = LineString(points_path2)

            # Compute frechet distance
            dist = frechet_distance(line1, line2)

            # Update if small enough distance 
            if dist <= self.frechet_threshold and dist < best_distance:
                best_distance = dist
                best_match = orphan_path
                best_idx = j

        if best_match:
            # If a match was found merge the paths 
            merged_path = self.merge_paths(path_to_check, best_match, self.batch_size)

            # Remove matched orphan path from orphan list
            self.orphan_paths.pop(best_idx)
            return merged_path, True

        return None, False


        # cost_array = []
        # for j, orphan_path in enumerate(self.orphan_paths):

        #     if len(path_to_check) < 2 or len(orphan_path) < 2:
        #         cost_array.append(np.inf)
        #         continue        # If single point, no path to compare
            
        #     # Separate points coordinates from frame numbers to compute frechet dist
        #     points_path1 = [t[0] for t in path_to_check]
        #     points_path2 = [t[0] for t in orphan_path]

        #     paths_dist = frechet_distance(LineString(points_path1), LineString(points_path2))
        #     cost_array.append(paths_dist)
    
        # potential_path = []
        # for j in range(len(self.orphan_paths)):
        #     if cost_array[j] <= self.frechet_threshold:
        #         potential_path.append((self.orphan_paths[j], cost_array[j]))

        # if len(potential_path) > 0:
        #     # Sort by Frechet distance
        #     potential_path.sort(key=lambda x:x[1])

        #     merged_path = self.merge_paths(path_to_check, potential_path[0][0], self.batch_size)

        #     return merged_path, True
        
        # else: 
        #     return None, False

    @staticmethod
    def merge_paths(path1, path2, batch_size):      #  path = [((x1, y1), f1), ((x2, y2), f2), ...] NOTE: should be sorted by frame numbers f1<f2<f3...

        # Sanity check that paths are ordered
        path1 = sorted(path1, key=lambda x: x[1])
        path2 = sorted(path2, key=lambda x: x[1])

        # Track paths with dict
        path1_dict = {frame: coords for coords, frame in path1}
        path2_dict = {frame: coords for coords, frame in path2}
        final_path = []

        for i in range(batch_size):
            if i in path1_dict and i in path2_dict:
                x1, y1 = path1_dict[i]
                x2, y2 = path2_dict[i]
                avg_coords = ((x1 + x2) / 2, (y1 + y2) / 2)
                final_path.append((avg_coords, i))
            elif i in path1_dict:
                final_path.append((path1_dict[i], i))
            elif i in path2_dict:
                final_path.append((path2_dict[i], i))
            else:
                continue  # Frame missing in both — optionally insert None

        return final_path

    def remove_duplicate_paths(self, merged_paths:list):
        """Remove near-duplicate paths (Fréchet distance too small)."""
        n = len(merged_paths)
        to_remove_indices = set()

        for i in range(n):
            if i in to_remove_indices or len(merged_paths[i]) < 2:
                continue
            points_i = [pt[0] for pt in merged_paths[i]]
            for j in range(i + 1, n):
                if j in to_remove_indices or len(merged_paths[j]) < 2:
                    continue
                points_j = [pt[0] for pt in merged_paths[j]]
                dist = frechet_distance(LineString(points_i), LineString(points_j))
                if dist <= self.similarity_threshold:
                    # Remove the shorter path
                    if len(merged_paths[i]) < len(merged_paths[j]):
                        to_remove_indices.add(i)
                        break  # Don't need to check i against others
                    else:
                        to_remove_indices.add(j)

        # Build filtered path list
        filtered_paths = [p for idx, p in enumerate(merged_paths) if idx not in to_remove_indices]
        return filtered_paths

        """previous version"""
        # paths_to_remove = []

        # # Check frechet distance between all paths
        # for checked_idx, path1 in enumerate(merged_paths):
        #     # Check path1 is not empty
        #     if len(path1) == 0:
        #         continue
        #     for path2 in merged_paths[(checked_idx + 1):]:
        #         # Check path2 is not empty
        #         if len(path2) == 0:
        #             continue
        #         if path1 == path2 or path2 in paths_to_remove: 
        #             continue

        #         # Separate points coordinates from frame numbers to compute frechet dist
        #         points_path1 = [t[0] for t in path1]
        #         points_path2 = [t[0] for t in path2]

        #         paths_dist = frechet_distance(LineString(points_path1), LineString(points_path2))

        #         if paths_dist <= self.similarity_threshold:
        #             if len(path1) < len(path2) and path1 not in paths_to_remove:
        #                 paths_to_remove.append(path1)
        #                 break
        #             else:
        #                 paths_to_remove.append(path2)

        # # Remove paths 
        # for path in paths_to_remove:
        #     merged_paths.remove(path)

        # return merged_paths

    def extend_global_paths(self, merged_paths, output_dir, overlap_frames):
        """Match global merged paths from new batch to current global paths using the Hunagrian algorithm"""
        existing_paths = self.global_batches_tracks
        num_existing = len(existing_paths)
        num_new = len(merged_paths)

        # Build cost matrix (shape: [num_existing, num_new])
        cost_matrix = np.full((num_existing, num_new), np.inf)
        for i, ex_path in enumerate(existing_paths):
            if len(ex_path) < 2:
                continue
            
            # Take last 10 points or all if fewer
            ex_traj = [p[0] for p in ex_path[-overlap_frames:]]

            # last point in case frechet fails
            ex_end = ex_path[-1][0]

            for j, new_path in enumerate(merged_paths):
                if len(new_path) < 2:
                    continue

                # Take first 10 points or all if fewer
                new_traj = [p[0] for p in new_path[:overlap_frames]]
                
                # First point in case frechet fails
                new_start = new_path[0][0]

                # Compute Fréchet distance
                try:
                    dist = frechet_distance(LineString(ex_traj), LineString(new_traj))
                    cost_matrix[i, j] = dist
                except Exception as e:
                    print(f"-----------Error computing Fréchet distance: {e}---------------\nUsing Euclidean distance instead.\n")
                    cost_matrix[i, j] = self.euclidean_distance(ex_end, new_start)
                    continue
                
        # Find rows and columns with at least one finite value
        valid_rows = np.any(np.isfinite(cost_matrix), axis=1)
        valid_cols = np.any(np.isfinite(cost_matrix), axis=0)

        valid_row_idx = np.where(valid_rows)[0]
        valid_col_idx = np.where(valid_cols)[0]

        if len(valid_row_idx) == 0 or len(valid_col_idx) == 0:
            raise ValueError("No valid assignments possible.")

        # Extract the valid submatrix
        reduced_cost_matrix = cost_matrix[np.ix_(valid_row_idx, valid_col_idx)]

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(reduced_cost_matrix)

        # Filter out assignments where cost is too high (optional)
        # For example, skip assignments above a certain threshold
        # threshold = 50.0
        # assignments = [(r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < threshold]

        # Merge paths in global_batches_tracks
        for r, c in zip(row_ind, col_ind):
            i = valid_row_idx[r]
            j = valid_col_idx[c]
            # NOTE: Could add a threshold here 

            # Debugging
            with open(f"{output_dir}\\paths_merging.txt", "a") as f:
                f.write(f"\n\nmerging existing path: \n {existing_paths[i]}\n\n and new path \n\n {merged_paths[j]}\n\n")
                
            existing_paths[i].extend(merged_paths[j])

        return existing_paths
            
    def handle_outliers(self, max_mvmt_between_frames, paths_by_cam):
        """Go through all paths and reassign points that were wrongly ided to the right path"""

        for cam_id, paths_list in paths_by_cam.items():
            outliers = []
            frames_present_in_paths = []

            # Extract outliers from each path without mutating during iteration
            for path in paths_list:
                frames_present = set()
                to_remove = []
                for i in range(len(path) - 1):
                    idx1 = i
                    while idx1 in to_remove: 
                        idx1 -= 1
                    point1 = path[idx1]
                    point2 = path[i + 1]
                    frames_present.add(point1[1])

                    if self.euclidean_distance(point1[0], point2[0]) > max_mvmt_between_frames:
                        outliers.append(point2)
                        to_remove.append(i + 1)
                    else:
                        frames_present.add(point2[1])
                
                # Remove outliers after iteration (avoid index shifting)
                for index in sorted(to_remove, reverse=True):
                    del path[index]

                frames_present_in_paths.append(frames_present)

            # Reassign outliers
            outliers.sort(key=lambda o: o[1])  # sort by frame number
            outliers_placed = []

            for outlier in outliers:
                if outlier in outliers_placed:
                    continue
                frame_out = outlier[1]
                coords_out = outlier[0]

                for path_idx, path in enumerate(paths_list):
                    frames = frames_present_in_paths[path_idx]
                    if frame_out in frames or len(path) <= 1:
                        continue

                    # Try inserting if adjacent frames match
                    coords_prev = next((p[0] for p in path if p[1] == frame_out - 1), None)
                    if coords_prev and self.euclidean_distance(coords_prev, coords_out) < max_mvmt_between_frames:
                        path.append(outlier)
                        path.sort(key=lambda x: x[1])  # maintain frame order
                        frames_present_in_paths[path_idx].add(frame_out)
                        outliers_placed.append(outlier)
                        break  # only assign to one path

        return paths_by_cam



        # for cam_id, paths_list in paths_by_cam.items():
        #     outliers = []
        #     frames_present_in_paths = []
        #     for path in paths_list:
        #         frames_present = []
        #         i = 0
        #         while i + 1 < len(path) and path[i + 1]:
        #             point1 = path[i]
        #             point2 = path[i + 1]
        #             frames_present.append(point1[1])

        #             # Check if distance between points is below threshold
        #             if self.euclidean_distance(point1[0], point2[0]) > max_mvmt_between_frames: 
        #                 outliers.append(point2)
        #                 path.remove(point2)
        #             else:
        #                 frames_present.append(point2[1])
        #                 i += 1
        #         frames_present_in_paths.append(set(frames_present))

        #     # Sort by euclidean distance
        #     outliers.sort(key=lambda outlier: outlier[1])
        #     outliers_placed = []
        #     for outlier in outliers:
        #         for path_idx, frames_path in enumerate(frames_present_in_paths):
        #             frame_number_outlied = outlier[1]
        #             coords_outlied = outlier[0]
        #             if frame_number_outlied not in frames_path and len(frames_path) > 1:

        #                 if frame_number_outlied - 1 >= 0 and frame_number_outlied - 1 < len(paths_list[path_idx]):
        #                     if self.euclidean_distance(coords_outlied, paths_list[path_idx][frame_number_outlied - 1][0]) < max_mvmt_between_frames:
        #                         paths_list[path_idx].insert(frame_number_outlied, outlier)
        #                         outliers_placed.append(outlier)
        #                         break

        # return paths_by_cam

    def merge_incomplete_paths(self, paths_list, batch_size, max_mvmt_pig):
        """Merge short paths that are close in time and space."""

        # Identify incomplete paths (shorter than 40% of batch)
        incomplete_indices = [i for i, path in enumerate(paths_list) if len(path) <= int(0.4 * batch_size)]
        completed_paths = [path for i, path in enumerate(paths_list) if i not in incomplete_indices]

        # Attempt to merge incomplete paths pairwise
        found_continuation = set()
        for i in incomplete_indices:
            if i in found_continuation: continue
            for j in incomplete_indices:
                if i == j:
                    continue

                path_i = paths_list[i]
                path_j = paths_list[j]

                if not path_i or not path_j:
                    continue

                end_i = path_i[-1]
                start_j = path_j[0]

                # Only merge if path_j is a continuation of path_i
                if end_i[1] < start_j[1] and self.euclidean_distance(end_i[0], start_j[0]) <= max_mvmt_pig:
                    merged = path_i + path_j
                    completed_paths.append(merged)
                    found_continuation.add(i)
                    # used.add(j)
                    break  

        # Remove leftover short paths (noise)
        return [path for path in completed_paths if len(path) > int(0.4 * batch_size)]            
        
    @staticmethod
    def euclidean_distance(p1, p2):
        """Calculate the Euclidean distance between two 2D points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def convert_to_builtin_type(self, obj):
        """Convert types from numpy types to regular python types"""
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
        
    def reinitialize_trackers_id_count(self):
        """Set all next_id variables to 1 in trackers"""
        for tracker in self.trackers:
            tracker.reinitialize_id_count()
        
        return