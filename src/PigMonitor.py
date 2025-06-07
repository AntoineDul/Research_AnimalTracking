import src.modules as modules
import src.config as config
import cv2
import os
from datetime import datetime

class PigMonitor:
    
    def __init__(self):
        self.detector = modules.Detector()
        self.mapper = modules.Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION)
        self.multi_tracker = modules.MultiTracker(num_cameras=config.NUM_CAMERAS, 
                                                  first_camera=config.FIRST_CAMERA, 
                                                  mapper=self.mapper, 
                                                  frechet_threshold=config.FRECHET_THRESHOLD,
                                                  similarity_threshold=config.SIMILARITY_THRESHOLD,
                                                  frechet_euclidean_weights= config.FRECHET_EUCLIDEAN_WEIGHTS,
                                                  batch_size=config.BATCH_SIZE,
                                                  overlapped_cams=config.CAM_FULLY_OVERLAPPED,
                                                  non_overlap_threshold=config.NON_OVERLAP_ZONES,
                                                  print_tracked=False,
                                                  )
        self.drawer = modules.Drawer()
        self.file_directory = config.MEDIAFLUX_VIDEO_DIR
        self.sync = modules.Synchronizer(cam_id_to_change=config.CAM_ID_TO_CHANGE, file_directory=self.file_directory)
        self.tracking_history_path = config.TRACKING_HISTORY_PATH
        self.frame_number = 0
        self.max_frames = 1000
    
    def process_batch_frame(self, frame, cam_id):
        # Undistort frame
        undistorted_frame = self.mapper.undistort_images(frame)

        # Detect pigs in the frame
        detections = self.detector.detect(undistorted_frame)

        # Update tracks 
        tracks = self.multi_tracker.track(detections, cam_id)

        return tracks

    def set_up_monitoring(self):
        """Processes all video files in the directory specified in config."""

        # Check if directory exists
        if not os.path.exists(self.file_directory):
            print(f"Error: Directory {self.file_directory} not found.")
            return
        
        # Get all video files in the directory
        files = os.listdir(self.file_directory)
        video_files = [f for f in files if f.endswith(('.mp3', '.avi', '.mp4'))]

        if not video_files:
            print("No video files found in the directory.")
            return

        # Retrieve and filter all video files
        sorted_cam_files = self.sync.separate_by_cameras(video_files)
        
        # Logs 
        with open(config.LOGS_PATH, "a") as f:
            f.write("\nThe following paths will be processed:\n")
            for cam_id, file_list in sorted_cam_files.items():
                f.write(f"\nCamera {cam_id}: {[data[1] for data in file_list]}\n")

        # Synchronize videos       
        self.sync.get_offsets()
        video_caps, _ = self.sync.synchronize()

        print("VIDEO CAPTURE OBJECTS INITIALIZED")
        print(video_caps)

        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Get video properties
        width = config.OUTPUT_VIDEO_WIDTH
        height = config.OUTPUT_VIDEO_HEIGHT
        fps = config.OUTPUT_VIDEO_FPS

        assert video_caps is not None

        return video_caps 

    def batch_monitor(self):
        """Main monitoring function, sets up and run monitoring. Saves all paths to a json file when finished running."""
        
        # Logs
        with open(config.LOGS_PATH, "w") as f:
            f.write("Beginning of tracking analysis.\n\n")

        self.multi_tracker.global_batches_tracks = [[] for _ in range(config.NUM_PIGS)]

        video_caps = self.set_up_monitoring()

        batch_count = 0
        monitor = True
        
        while monitor:

            # Initialize individual cams paths tracking 
            paths = {}
            
            for i in range(config.NUM_CAMERAS):
                paths[i + config.FIRST_CAMERA] = [[] for _ in range(config.NUM_PIGS + 10)]  # + 10 to leave room for wrong re-ids during batch, they are handled later

            # Set all id tracking to 1 => to avoid getting huge id numbers after a great nb of batches due to individual cam re-ids
            self.multi_tracker.reinitialize_trackers_id_count()

            # Clear unmatched paths from previous batch
            self.multi_tracker.orphan_paths = []

            all_successful = True

            for batch_frame_number in range(config.BATCH_SIZE):

                for cam_id, cap in video_caps.items():
                    success, frame = cap.read()

                    # If no more frame in one of the videos we are processing, try to start the process of the next one 
                    if not success:

                        # Logs
                        with open(config.LOGS_PATH, "a") as f:
                            f.write(f"Camera {cam_id} has ran out of frames, trying to start next video.")

                        cap.release()
                        cap = self.sync.start_next_video(cam_id)

                        # If no more videos to process, break
                        if cap is None: 
                            print(f"Camera {cam_id} has no more frames.")       
                            all_successful = False
                            break

                        # Else update the cap object of the corresponding camera
                        else:
                            video_caps[cam_id] = cap
                            success, frame = cap.read()

                    # Process frames
                    tracks = self.process_batch_frame(frame, cam_id)

                    for track in tracks:
                        paths[cam_id][track['id']].append((track['center'], batch_frame_number))


                if not all_successful:
                    break
            
            # Correct bias and handle outliers in the detected paths 
            unbiased_paths = self.mapper.fix_paths_bias(paths, config.THALES_SCALE, config.CAM_POSITIONS)

            # Relocate the outliers in the paths in their correct path or discard them if they were noise
            rebuilt_paths = self.multi_tracker.handle_outliers(config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES, unbiased_paths, config.LOGS_PATH) 

            # Merge pieces of the same paths that were separated and discard the paths that are too short 
            clean_paths = {cam_id: self.multi_tracker.merge_incomplete_paths(paths_list, config.BATCH_SIZE, config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES, logs_path=config.LOGS_PATH) for cam_id, paths_list in rebuilt_paths.items()}

            # Merge paths together sequentially
            paths_8_17 = self.multi_tracker.batch_match(clean_paths[8], clean_paths[9], 8, 17, config.LOGS_PATH)  # Match batch paths across cameras
            paths_8_7_17 = self.multi_tracker.batch_match(clean_paths[7], paths_8_17, 7, None, config.LOGS_PATH)
            paths_5_7_8_17 = self.multi_tracker.batch_match(clean_paths[5], paths_8_7_17, 5, None, config.LOGS_PATH)
            all_paths_merged = self.multi_tracker.batch_match(clean_paths[6], paths_5_7_8_17, 6, None, config.LOGS_PATH)

            # Last check for duplicate paths
            final_paths_merged = self.multi_tracker.remove_duplicate_paths(all_paths_merged, config.NUM_PIGS)
            
            # Logs
            with open(config.LOGS_PATH, "a") as f:
                f.write(f"\n\n----- {len(final_paths_merged)} PATHS GLOBALLY -----\n\n")
                for i, path in enumerate(final_paths_merged):
                    f.write(f"\n\nPath {i} (len {len(path)}): {path}\n\n")
                f.write("---------------------------------------------")
            
            # need to make sure this extends right path
            if batch_count == 0:
                for idx, path in enumerate(final_paths_merged):
                    self.multi_tracker.global_batches_tracks[idx].extend(path)
            else:
                self.multi_tracker.extend_global_paths(
                    merged_paths=final_paths_merged, 
                    logs_path=config.LOGS_PATH, 
                    overlap_frames=config.REWIND_FRAMES,
                    batch_size=config.BATCH_SIZE,
                    batch_count=batch_count,
                    )                

            # Rewind a number of frames to overlap with next batch
            for cam_id, cap in video_caps.items():
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - config.REWIND_FRAMES))

            # Increment batch count
            batch_count += 1
        
            # Plot the paths detected in the batch for each pov for the first 10 batches
            if batch_count <= 9:
                for i in [5, 6, 7, 8, 9]:
                    self.drawer.plot_batch_paths(clean_paths[i], i, config.BATCH_SIZE, config.BATCH_PLOTS_PATH, batch_count) 
                labelled_paths = [("8_17", paths_8_17), ("8_7_17", paths_8_7_17), ("5_7_8_17", paths_5_7_8_17)]
                for label, paths in labelled_paths:
                    self.drawer.plot_batch_paths(paths, label, config.BATCH_SIZE, config.BATCH_PLOTS_PATH, batch_count)

            # Plot overall paths so far 
            self.drawer.plot_batch_paths(self.multi_tracker.global_batches_tracks, "overall", config.BATCH_SIZE, config.BATCH_PLOTS_PATH, batch_count)
            
            # Save tracking history to json file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{config.TRACKING_HISTORY_PATH}\\{timestamp}_track_{config.BATCH_SIZE}_batch-size_{batch_count}_batches.json"
            self.multi_tracker.save_tracking_history(file_path=save_path)
            
            print(f"\n\n======================= batch {batch_count} complete. ===================================\n\n")
            with open(config.LOGS_PATH, "a") as f:
                f.write(f"\n\n======================= batch {batch_count} complete. ===================================\n\n")
            
            # Monitoring for 200 batches of 200 frames with 20 frames overlap, i.e. 180 new frames per batch (9 seconds) => total 30 minutes of monitoring.
            if batch_count == 200:
                monitor=False

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()

# Run the pipeline
if __name__ == "__main__":
    monitor = PigMonitor()
    monitor.batch_monitor()