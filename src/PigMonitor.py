import src.modules as modules
import src.config as config
import numpy as np
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
                                                  cluster_eps=config.CLUSTER_EPSILON, 
                                                  cluster_min_samples=config.CLUSTER_MIN_SAMPLES, 
                                                  max_age=config.MAX_GLOBAL_AGE,
                                                  max_cluster_distance= config.MAX_CLUSTER_DISTANCE, 
                                                  frechet_threshold=config.FRECHET_THRESHOLD,
                                                  similarity_threshold=config.SIMILARITY_THRESHOLD,
                                                  batch_size=config.BATCH_SIZE,
                                                  overlapped_cams=config.CAM_FULLY_OVERLAPPED,
                                                  non_overlap_threshold=config.NON_OVERLAP_ZONES,
                                                  print_tracked=False,
                                                  )
        self.drawer = modules.Drawer()
        self.file_directory = config.MEDIAFLUX_VIDEO_DIR
        self.sync = modules.Synchronizer(self.file_directory)
        self.video_writer_path = config.PROCESSED_VIDEO_PATH
        self.tracking_history_path = config.TRACKING_HISTORY_PATH
        self.output_plot_path = config.OUTPUT_PLOT_PATH
        self.first_camera = 5
        self.frame_number = 0
        self.max_frames = 1000


    def process_frame(self, frame, frame_count, cam_id=None):
        # Undistort frame
        undistorted_frame = self.mapper.undistort_images(frame)

        # Detect pigs in the frame
        detections = self.detector.detect(undistorted_frame)

        # Update tracks 
        tracks = self.multi_tracker.track(detections, cam_id)

        # Draw tracks on the frame
        display_frame = self.drawer.draw_bboxes(undistorted_frame.copy(), tracks)
        display_frame = self.drawer.add_useful_info(display_frame, frame_count, tracks)

        return display_frame, tracks
    
    def process_batch_frame(self, frame, frame_count, cam_id):
        # Undistort frame
        undistorted_frame = self.mapper.undistort_images(frame)

        # Detect pigs in the frame
        detections = self.detector.detect(undistorted_frame)

        # Update tracks 
        tracks = self.multi_tracker.track(detections, cam_id)

        return tracks

    def monitor(self, video_path):
        """Processes a single video file."""

        # Check if video path exists
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
        else:
            print(f"Error: Video file {video_path} not found.")
            return
        
        # Check if video capture opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if not available
        
        out = cv2.VideoWriter(self.video_writer_path, fourcc, fps, (width, height))
        
        print("Press 'q' to quit.")
        print("Press 's' to save a screenshot.")
        print("Press '+' to increase detection threshold.")
        print("Press '-' to decrease detection threshold.")

        frame_count = 0
        screenshot_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to improve performance (process every 2nd frame)
            if frame_count % config.FRAME_SKIP != 0 and frame_count > 1:
                continue
                
            # Process frame
            processed_frame, tracks = self.process_frame(frame, frame_count)
            
            # Write to output video
            out.write(processed_frame)
                
            # Resize frame for display
            display_frame = cv2.resize(processed_frame, (1600, 900))     
                
            # Display frame
            cv2.imshow("Pig Tracking", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save a screenshot
                screenshot_path = f"outputs/screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"Saved screenshot to {screenshot_path}")
                screenshot_count += 1
            elif key == ord('+'):
                # Increase threshold
                self.detector.increase_threshold()
                print(f"Detection threshold increased to {self.detector.get_confidence_threshold():.2f}")
            elif key == ord('-'):
                # Decrease threshold
                self.detector.decrease_threshold()
                print(f"Detection threshold increased to {self.detector.get_confidence_threshold():.2f}")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking complete. Output saved to {self.video_writer_path}")

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

        video_caps = {}

        # Synchronize videos
        sorted_videos = self.sync.separate_by_cameras(video_files)
        camera_offsets = self.sync.get_offsets()
        video_caps, _ = self.sync.synchronize(sorted_videos, camera_offsets)
        
        print("VIDEO CAPTURE OBJECTS INITIALIZED")
        print(video_caps)

        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Get video properties
        width = config.OUTPUT_VIDEO_WIDTH
        height = config.OUTPUT_VIDEO_HEIGHT
        fps = config.OUTPUT_VIDEO_FPS
        out = cv2.VideoWriter(self.video_writer_path, fourcc, fps, (width, height))

        assert video_caps is not None
        assert out is not None

        return video_caps, out

    def multi_monitor(self):

        video_caps, out = self.set_up_monitoring()

        frame_count = 0

        # Frame sync + display loop
        while True:
            frames = {}
            all_successful = True

            frame_count += 1
            # Skip frames to improve performance (process every 2nd frame)
            if frame_count % config.FRAME_SKIP != 0 and frame_count > 1:
                continue

            for cam_id, cap in video_caps.items():
                success, frame = cap.read()

                if not success:
                    print(f"Camera {cam_id} has no more frames.")       # TODO : implement queue logic to handle next video
                    all_successful = False
                    break

                # Process frame
                processed_frame, _ = self.process_frame(frame, frame_count, cam_id)

                frames[cam_id] = processed_frame

            if not all_successful:
                break

            self.multi_tracker.globally_match_tracks()  # Match tracks across cameras

            grid = self.drawer.make_grid(frames)
            out.write(grid)
            cv2.imshow('Synchronized 2x2 Grid', grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.multi_tracker.save_tracking_history(self.tracking_history_path)
        self.drawer.plot_logs(self.tracking_history_path, self.output_plot_path)

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()

    def batch_monitor(self):
        
        self.multi_tracker.global_batches_tracks = [[] for _ in range(config.NUM_PIGS + 10)]

        video_caps, out = self.set_up_monitoring()
        with open(f"{config.OUTPUT_DIR}\\paths_merging.txt", "w") as f:
            f.write("Beginning of merge analysis.\n\n")

        batch_count = 0
        frame_count = 0

        monitor = True
        
        while monitor:

            # Initialize individual cams paths tracking 
            paths = {}
            
            for i in range(config.NUM_CAMERAS):
                paths[i + self.first_camera] = [[] for _ in range(config.NUM_PIGS + 10)]  # + 10 to leave room for wrong re-ids during batch, they are handled later

            # Set all id tracking to 1 => to avoid getting huge id numbers after a great nb of batches due to individual cam re-ids
            self.multi_tracker.reinitialize_trackers_id_count()

            # Clear unmatched paths from previous batch
            self.multi_tracker.orphan_paths = []

            all_successful = True
            frame_count += 1

            # Skip frames to improve performance (process every 2nd frame)
            if frame_count % config.FRAME_SKIP != 0 and frame_count > 1:
                continue

            for batch_frame_number in range(config.BATCH_SIZE):

                for cam_id, cap in video_caps.items():
                    success, frame = cap.read()

                    if not success:
                        print(f"Camera {cam_id} has no more frames.")       # TODO : implement queue logic to handle next video
                        all_successful = False
                        break

                    # Process frames
                    tracks = self.process_batch_frame(frame, frame_count, cam_id)

                    for track in tracks:
                        paths[cam_id][track['id']].append((track['center'], batch_frame_number))


                if not all_successful:
                    break
            
            # Correct bias and handle outliers in the detected paths 
            unbiased_paths = self.mapper.fix_paths_bias(paths, config.THALES_SCALE, config.CAM_POSITIONS)

            # Relocate the outliers in the paths in their correct path or discard them if they were noise
            rebuilt_paths = self.multi_tracker.handle_outliers(config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES, unbiased_paths) 

            # Merge pieces of the same paths that were separated and discard the paths that are too short 
            clean_paths = {cam_id: self.multi_tracker.merge_incomplete_paths(paths_list, config.BATCH_SIZE, config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES) for cam_id, paths_list in rebuilt_paths.items()}

            # --- Debugging ---
            # for cam_id in [5, 6, 7, 8, 9]:
            #     print(f"========== cam {cam_id} ===========")
            #     for i, path in enumerate(clean_paths[cam_id]):
            #         print(f"path {i} (length {len(path)}): {path}")
            #     print("=====================================")

            # Merge paths together sequentially
            paths_7_17 = self.multi_tracker.batch_match(clean_paths[7], clean_paths[9], 7, 17)  # Match batch paths across cameras
            paths_8_7_17 = self.multi_tracker.batch_match(clean_paths[8], paths_7_17, 8, None)
            paths_5_7_8_17 = self.multi_tracker.batch_match(clean_paths[5], paths_8_7_17, 5, None)
            all_paths_merged = self.multi_tracker.batch_match(clean_paths[6], paths_5_7_8_17, 6, None)

            # Concatenate or discard pieces of paths together 
            # fused_paths = self.multi_tracker.merge_incomplete_paths(all_paths_merged, config.BATCH_SIZE)

            # Last check for duplicate paths
            final_paths_merged = self.multi_tracker.remove_duplicate_paths(all_paths_merged)
            
            # need to make sure this extends right path
            if batch_count == 0:
                for idx, path in enumerate(final_paths_merged):
                    # if idx == 0 : continue
                    self.multi_tracker.global_batches_tracks[idx].extend(path)
            else:
                self.multi_tracker.extend_global_paths(merged_paths=final_paths_merged, output_dir=config.OUTPUT_DIR, overlap_frames=config.REWIND_FRAMES)                

            # Rewind a number of frames to overlap with next batch
            for cam_id, cap in video_caps.items():
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - config.REWIND_FRAMES))

            # Increment batch count
            batch_count += 1
        
            # Plot the paths detected in the batch for each pov
            for i in [5, 6, 7, 8, 9]:
                self.drawer.plot_batch_paths(clean_paths[i], i, config.BATCH_SIZE, config.BATCH_PLOTS_PATH, batch_count) 

            # Plot overall paths so far 
            self.drawer.plot_batch_paths(self.multi_tracker.global_batches_tracks, "overall", config.BATCH_SIZE, config.BATCH_PLOTS_PATH, batch_count)
            
            print(f"\n\n======================= batch {batch_count} complete. ===================================\n\n")
            with open(f"{config.OUTPUT_DIR}\\paths_merging.txt", "a") as f:
                f.write(f"\n\n======================= batch {batch_count} complete. ===================================\n\n")
            
            # Debugging
            if batch_count == 12:
                monitor=False
                
        # --- Debugging ---
        # for i, path in enumerate(self.multi_tracker.global_batches_tracks):
        #     print(f"\n\npath {i} (length {len(path)}): {path}\n\n")

        # Save tracking history to json file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{config.OUTPUT_DIR}\\track_history\\{timestamp}_track_{config.BATCH_SIZE}_batch-size_{batch_count}_batches.json"
        self.multi_tracker.save_tracking_history(file_path=save_path)

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()



