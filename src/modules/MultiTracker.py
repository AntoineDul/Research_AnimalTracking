import numpy as np
import math
from shapely import LineString, frechet_distance
import json
from src.modules.Tracker import Tracker

class MultiTracker:

    def __init__(self, num_cameras=5, first_camera=5, mapper=None,  frechet_threshold=3, similarity_threshold=0.05, frechet_euclidean_weights=[1.0, 0.5], batch_size=10, overlapped_cams=[5, 6, 7], non_overlap_threshold=[-1.5, 1], print_tracked=False):
        self.trackers = [Tracker() for _ in range(num_cameras)]
        self.mapper = mapper    
        self.num_cameras = num_cameras
        self.first_camera = first_camera
        self.overlapped_cams = overlapped_cams

        # tracker parameters
        # self.max_age = max_age                                   # Maximum age of a track before it is removed -> might remove later
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
        self.max_path_length_ratio = 0.3                         # Max lengths ratio acceptable to match paths (avoid matching 2 points with a 20 points path)
        self.frechet_euclidean_weights = frechet_euclidean_weights

    def save_tracking_history(self, file_path):
        """Save the tracking history to a JSON file."""

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
           
# =============== Batch implementation ==============================
    def batch_match(self, paths_list1, paths_list2, cam_id1, cam_id2, logs_path):  # paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        """
        Match and merge similar paths from two different camera views.
        paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        """
        
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
                
                # Get the common frame window to compare paths
                start = max(path1[0][1], path2[0][1])
                end = min(path1[-1][1], path2[-1][1])
                if start >= end:
                    continue    # no overlap 

                # Separate points coordinates from frame numbers to compute frechet dist
                coords1 = [p[0] for p in path1 if start <= p[1] <= end]
                coords2 = [p[0] for p in path2 if start <= p[1] <= end]

                if len(coords1) < 2 or len(coords2) < 2:
                    continue  # Not enough points to compare

                # Compute Frechet distance between paths and add the min distance between endpoints
                paths_dist = frechet_distance(LineString(coords1), LineString(coords2)) + min(self.euclidean_distance(coords1[0], coords2[0]), self.euclidean_distance(coords2[-1], coords1[-1]))

                # Fill cost matrix
                cost_matrix[i, j] = paths_dist

        # Check if cost matrix is not empty
        if np.all(np.isinf(cost_matrix)):
            if empty_paths1 == len_paths_1 and empty_paths2 == len_paths_2:
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nNo paths in either cam {cam_id1} and {cam_id2} \n\n")
                return paths_list1
            elif empty_paths1 == len_paths_1:
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nNo paths in cam {cam_id1}\n\n")
                return paths_list2
            elif empty_paths2 == len_paths_2:
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nNo paths in cam {cam_id2} \n\n")
                return paths_list1
            else: 
                raise ValueError

        # Sort all path pairs by frechet distance
        best_paths_pairs = []
        for i in range(len(paths_list1)):
            for j in range(len(paths_list2)):
                if cost_matrix[i, j] >= 0 and cost_matrix[i, j] <= self.frechet_threshold:
                    best_paths_pairs.append((i, j, cost_matrix[i, j]))
                else:

                    # Logs
                    with open(logs_path, "a") as f:
                        f.write(f"\n\nNot adding paths\n {paths_list1[i]} and path \n{paths_list2[j]} \n to the best pairs, cost is {cost_matrix[i, j]}\n\n")
        
        # Sort by frechet distance 
        best_paths_pairs.sort(key=lambda x: x[2])

        # Track assignments
        assigned_paths1 = set()
        assigned_paths2 = set()

        # Greedy assignment of paths
        for path1_idx, path2_idx, _ in best_paths_pairs:
            if path1_idx not in assigned_paths1 and path2_idx not in assigned_paths2:
                
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nBest frechet dist found between path\n {paths_list1[i]}\n and \n{paths_list2[j]}\n is \n{cost_matrix[i, j]}. Merging and adding path. \n\n")
                
                # Merge and add path to global tracks
                merged_path = self.merge_paths(paths_list1[path1_idx], paths_list2[path2_idx], self.batch_size)
                global_paths.append(merged_path)
                assigned_paths1.add(path1_idx)
                assigned_paths2.add(path2_idx)
                
        # Function to check if a pig is in a zone covered by only one cam 
        def is_outside_overlap(path):
            avg_y = (path[0][0][1] + path[-1][0][1]) / 2
            return avg_y <= self.non_overlap_threshold[0] or (avg_y >= self.non_overlap_threshold[1])

        # Iterate through the remaining non assigned paths
        for i, path in enumerate(paths_list1):
            if i not in assigned_paths1:
                # Add paths if it is in a zone covered only by that camera
                if cam_id1 not in self.overlapped_cams and is_outside_overlap(path):
                    # Logs
                    with open(logs_path, "a") as f:
                        f.write(f"\n\nPath:\n {path}\n no match found but is in non-overlapped zone, so adding it. \n\n")
                    global_paths.append(path)

                # If paths_list1  is already the combination of 2 povs
                elif cam_id1 is None:
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
                if cam_id2 not in self.overlapped_cams and is_outside_overlap(path):
                    # Logs
                    with open(logs_path, "a") as f:
                        f.write(f"\n\nPath:\n {path}\n no match found but is in non-overlapped zone, so adding it. \n\n")
                    global_paths.append(path)

                # If path_list2 is already the combination of 2 povs
                elif cam_id2 is None:
                    global_paths.append(path)

                else:
                    merged_path, found = self.check_orphan_paths(path)
                    if found:
                        global_paths.append(merged_path)
                    else:
                        self.orphan_paths.append(path)

        # Logs
        with open(logs_path, "a") as f:
            f.write(f"\n\nPaths AFTER batch_match between cam {cam_id1} and {cam_id2} ({len(global_paths)} paths): \n\n")
            for i, path in enumerate(global_paths):
                f.write(f"\npath {i}: {path}\n")

        return global_paths

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
                continue  # Frame missing in both 

        return final_path

    def remove_duplicate_paths(self, merged_paths:list, total_nb_pig):
        """Select spatially unique, dynamic, and well-separated paths."""

        filtered = []
        for path in merged_paths:
            # remove path that have the same values for all their x or y coordinates
            if len(path) < 2:
                continue
            xs = [p[0][0] for p in path]
            ys = [p[0][1] for p in path]
            if len(set(xs)) > 1 and len(set(ys)) > 1:
                filtered.append(path)

        scores = []
        avg_positions = []

        for i, path_i in enumerate(filtered):
            if len(path_i) < 2:
                continue

            points_i = [pt[0] for pt in path_i]
            avg_pos_i = np.mean(points_i, axis=0)
            avg_positions.append((i, avg_pos_i))

        for i, path_i in enumerate(filtered):
            if len(path_i) < 2:
                continue

            points_i = [pt[0] for pt in path_i]
            line_i = LineString(points_i)

            # 1. Minimum Frechet distance to other paths
            min_frechet = float("inf")
            for j, path_j in enumerate(filtered):
                if i == j or len(path_j) < 2:
                    continue
                points_j = [pt[0] for pt in path_j]
                line_j = LineString(points_j)
                dist = frechet_distance(line_i, line_j)
                min_frechet = min(min_frechet, dist)

            # 2. Average movement in the path
            avg_euclidean = np.mean([
                self.euclidean_distance(points_i[k], points_i[k + 1])
                for k in range(len(points_i) - 1)
            ])

            # 3. Minimum average position distance to other paths
            avg_pos_i = avg_positions[i][1]
            min_avg_pos_dist = min([
                self.euclidean_distance(avg_pos_i, avg_positions[j][1])
                for j in range(len(avg_positions)) if j != i
            ])

            # Weighted score (you can tune the weights)
            score = (
                0.8 * min_frechet + 
                0 * avg_euclidean + 
                0.2 * min_avg_pos_dist
            )
            scores.append((i, score))

        # Sort and keep the best
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:total_nb_pig]]
        return [filtered[i] for i in top_indices]

    def extend_global_paths(self, merged_paths, logs_path, overlap_frames, batch_size, batch_count):
        """Match global merged paths from new batch to current global paths"""

        existing_paths = self.global_batches_tracks
        num_existing = len(existing_paths)
        num_new = len(merged_paths)

        # Build cost matrix (shape: [num_existing, num_new])
        cost_matrix = np.full((num_existing, num_new), np.inf)
        for i, ex_path in enumerate(existing_paths):
            if len(ex_path) < 2:
                continue
            
            # Take last 10 points or all if fewer
            ex_traj = [p[0] for p in ex_path[-overlap_frames:] if p[1] >= batch_size - (2 * overlap_frames)]    # Times 2 because we already shifted frames by minus overlap frames when extending global paths

            # last point in case frechet fails
            ex_end = ex_path[-1][0]

            for j, new_path in enumerate(merged_paths):
                if len(new_path) < 2:
                    continue

                # Take all points in the overlap
                new_traj = [p[0] for p in new_path[:overlap_frames] if p[1] < overlap_frames]
                
                # First point in case frechet fails (no overlap)
                new_start = new_path[0][0]
            
                min_len = min(len(ex_traj), len(new_traj))

                # If there are indeed some points in the overlap
                if min_len >= 2:

                    # Compute Fréchet distance
                    frechet_dist = frechet_distance(LineString(ex_traj), LineString(new_traj))

                    # Compute average Euclidean distance
                    avg_euc = np.mean([self.euclidean_distance(ex_traj[k], new_traj[k]) for k in range(min_len)])

                    # Combine into a final cost
                    total_cost = self.frechet_euclidean_weights['Frechet'] * frechet_dist + self.frechet_euclidean_weights['Euclidean'] * avg_euc  
                
                # If no overlap, compare euclidean distance
                else:
                    total_cost = self.euclidean_distance(ex_end, new_start)

                cost_matrix[i, j] = total_cost

                
        # Sort all path pairs by cost
        best_paths_pairs = []
        for i in range(num_existing):
            for j in range(num_new):

                #remove later 
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nCost between existing path\n {existing_paths[i]}\n and new path \n{merged_paths[j]}\n is : {cost_matrix[i, j]}. \n\n")
        
                if cost_matrix[i, j] >= 0 and cost_matrix[i, j] < 5:
                    best_paths_pairs.append((i, j, cost_matrix[i, j]))
        
        # Sort by frechet distance 
        best_paths_pairs.sort(key=lambda x: x[2])

        # Track assignments
        assigned_ex = set()
        assigned_new = set()
        assigned_count = 0

        # Correct frame numbers
        new_merged_paths = []
        for i, path in enumerate(merged_paths):
            new_path = []
            for point in path:
                frame_nb = point[1]
                if frame_nb >= overlap_frames:
                    corrected_frame = batch_size * batch_count + point[1] - overlap_frames
                    new_path.append((point[0], corrected_frame))
            new_merged_paths.append(new_path)
        
        # Sanity check
        assert len(new_merged_paths) == len(merged_paths)

        # Extend paths with lowest cost 
        for ex_idx, new_idx, _ in best_paths_pairs:
            if ex_idx not in assigned_ex and new_idx not in assigned_new:
                
                # Logs
                with open(logs_path, "a") as f:
                    f.write(f"\n\nMerging existing path\n {existing_paths[ex_idx]}\n and \n{new_merged_paths[new_idx]}\n Cost is : {cost_matrix[ex_idx, new_idx]}. \n\n")

                # Merge and add path to global tracks
                existing_paths[ex_idx].extend(new_merged_paths[new_idx])
                assigned_ex.add(ex_idx)
                assigned_new.add(new_idx)
                assigned_count += 1
                
                # Only one assignment per path
                if assigned_count == len(new_merged_paths):
                    break

        return existing_paths
            
    def handle_outliers(self, max_mvmt_between_frames, paths_by_cam:dict, logs_path):
        """Go through all paths and reassign points that were wrongly ided to the right path"""
        
        # Logs
        if logs_path is not None:
            with open(logs_path, "a") as f:
                f.write("-------------- handle outliers analysis -----------------")
                f.write(f"\npaths before handling outliers:\n")
                for cam_id, paths_list in paths_by_cam.items():
                    f.write(f"=== cam {cam_id} ===")
                    for i, path in enumerate(paths_list):
                        f.write(f"\npath {i}: {path}\n")

        for cam_id, paths_list in paths_by_cam.items():
            # print(f"cam {cam_id}\npaths_list:{paths_list}")
            outliers = []
            frames_present_in_paths = []

            # Extract outliers from each path without mutating during iteration
            for path in paths_list:
                
                removed_in_a_row = 0
                frames_present = set()
                to_remove = []
                for i in range(len(path) - 1):
                    idx1 = i
                    while idx1 in to_remove: 
                        idx1 -= 1
                    point1 = path[idx1]
                    point2 = path[i + 1]

                    frames_present.add(point1[1])

                    if self.euclidean_distance(point1[0], point2[0]) > max_mvmt_between_frames or removed_in_a_row > 50:
                        outliers.append(point2)
                        to_remove.append(i + 1)
                        removed_in_a_row += 1
                    else:
                        frames_present.add(point2[1])
                        removed_in_a_row = 0
                
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

            # Check if point is in pen boundaries, if not shift it into them
            for idx, path in enumerate(paths_list):
                if len(path) == 0:
                    continue

                for i, point in enumerate(path):
                    point = self.check_point_in_boundaries(point)
                    path[i] = point
                        
        # Logs 
        if logs_path is not None:
            with open(logs_path, "a") as f:
                f.write(f"\n\n\npaths AFTER handling outliers:\n")
                for cam_id, paths_list in paths_by_cam.items():
                    f.write(f"=== cam {cam_id} ===")
                    for i, path in enumerate(paths_list):
                        f.write(f"\npath {i}: {path}\n")
                f.write("------------------------------------------------")
                
        return paths_by_cam

    def merge_incomplete_paths(self, paths_list, batch_size, max_mvmt_pig, logs_path):
        """Merge short paths that are close in time and space."""
                
        # Logs
        with open(logs_path, "a") as f:
            f.write(f"\n\nPaths before merge_incomplete_paths ({len(paths_list)} paths): \n\n")
            for i, path in enumerate(paths_list):
                f.write(f"\npath {i}: {path}\n")

        found_continuation = {i: None for i in range(len(paths_list))}  # {path index in paths_list : continuation path idx in paths_list}
        found_preceding = {i: None for i in range(len(paths_list))}     # {path index in paths_list : continuation path idx in paths_list}

        for i, path_i in enumerate(paths_list):
            if found_continuation[i] != None or not path_i: 
                continue
  
            for j, path_j in enumerate(paths_list):
                if i == j or not path_j or found_preceding[j] != None:
                    continue

                end_i = path_i[-1]
                start_j = path_j[0]

                # Only merge if path_j is a continuation of path_i
                if (
                end_i[1] < start_j[1]                                                 # frame number of end of path_i is smaller than frame number of start of path_j
                and start_j[1] - end_i[1] < 0.3 * batch_size                          # the time difference between the two paths is not too big
                and self.euclidean_distance(end_i[0], start_j[0]) <= max_mvmt_pig     # euclidean distance between end and start of the two paths is small enough
                and len(path_i) + len(path_j) <= batch_size):                         # the combined size of the paths is not bigger than batch size 
                    
                    found_continuation[i] = j 
                    found_preceding[j] = i
                    break 

        final_merged_paths = []
        visited = set()
        
        for idx in range(len(paths_list)):
            if idx in visited or not paths_list[idx]: 
                continue
            path = paths_list[idx].copy()
            visited.add(idx)

            # Forward merge
            continuation = found_continuation[idx]
            while continuation is not None and continuation not in visited:
                next_continuation = found_continuation[continuation]
                path += paths_list[continuation]
                visited.add(continuation)
                continuation = next_continuation

            # Backward merge
            preceding = found_preceding[idx]
            while preceding is not None and preceding not in visited:
                next_preceding = found_preceding[preceding]
                path = paths_list[preceding] + path
                visited.add(preceding)
                preceding = next_preceding
            
            if ((path is not None) and (len(path) > 0.4 * batch_size)):
                final_merged_paths.append(path)
                
        # Logs
        with open(logs_path, "a") as f:
            f.write(f"\n\nPaths AFTER merge_incomplete_paths ({len(final_merged_paths)} paths): \n\n")
            for i, path in enumerate(final_merged_paths):
                f.write(f"\npath {i}: {path}\n")
        
        return final_merged_paths
        
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
    
    @staticmethod
    def check_point_in_boundaries(original_point):
        coords = list(original_point[0])  # (x, y)
        frame = original_point[1]

        # Clamp x and y
        x = max(0, min(2.5, coords[0]))
        y = max(-3, min(3, coords[1]))

        return ((x, y), frame)