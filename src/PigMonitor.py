import src.modules as modules
import src.config as config
import cv2
import os

class PigMonitor:
    
    def __init__(self):
        self.detector = modules.Detector()
        self.mapper = modules.Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION)
        self.multi_tracker = modules.MultiTracker(config.NUM_CAMERAS, config.FIRST_CAMERA, self.mapper, config.CLUSTER_EPSILON, config.CLUSTER_MIN_SAMPLES, config.MAX_GLOBAL_AGE, config.MAX_CLUSTER_DISTANCE, False)
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
        
        self.multi_tracker.global_batches_tracks = [[] for _ in range(config.NUM_PIGS)]

        video_caps, out = self.set_up_monitoring()

        batch_count = 0
        frame_count = 0

        a = True
        
        while a:
            paths = {}

            for i in range(config.NUM_CAMERAS):
                paths[i + self.first_camera] = [[] for _ in range(config.NUM_PIGS)]

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

                    # print(f"\ntracks for cam {cam_id}: {tracks}\n")
                    # print(f"iteration tracks: {tracks}")

                    for track in tracks:
                        paths[cam_id][track['id']].append((track['center'], batch_frame_number))


                if not all_successful:
                    break
            
            # print("-----------------BATCH SUMMARY (before merging)--------------------")
            for cam_id, batch in paths.items():
                paths[cam_id] = self.mapper.batch_to_world_coords(batch, cam_id)           
                
                # print(f"\n\nCAMERA {cam_id} // \n ORIGINAL BATCH: {batch} \n\n GLOBAL COORDS BATCH: {paths[cam_id]}")

            # print("--------------------------------------------------------------")

            clean_paths = self.multi_tracker.handle_outliers(config.MAX_PIG_MVMT_BETWEEN_TWO_FRAMES, paths) 

            paths_7_17 = self.multi_tracker.batch_match(clean_paths[7], clean_paths[9], 7, 17)  # Match batch paths across cameras
            paths_8_7_17 = self.multi_tracker.batch_match(clean_paths[8], paths_7_17, 8, None)
            paths_5_7_8_17 = self.multi_tracker.batch_match(clean_paths[5], paths_8_7_17, 5, None)
            all_paths_merged = self.multi_tracker.batch_match(clean_paths[6], paths_5_7_8_17, 6, None)

            # print(f"\n\nALL PATHS MERGED:")
            # print(all_paths_merged)
            # print(f"global batches tracks: \n{self.multi_tracker.global_batches_tracks}")
            
            # need to make sure this extends right path
            # TODO inaccuracies here 
            for idx, path in enumerate(all_paths_merged):
                if idx == 0: continue
                if batch_count == 0:
                    self.multi_tracker.global_batches_tracks[idx].extend(path)
                else:
                    placed = False
                    dist_to_closest_endpoint = 1000
                    best_path_idx = None
                    possible_indices = [i for i in range(len(self.multi_tracker.global_batches_tracks))]
                    for track_idx in possible_indices:
                        if len(self.multi_tracker.global_batches_tracks[track_idx]) > 0:
                            distance = self.multi_tracker.euclidean_distance(path[0][0], self.multi_tracker.global_batches_tracks[track_idx][-1][0])
                            if distance < dist_to_closest_endpoint:
                                dist_to_closest_endpoint = distance
                                best_path_idx = track_idx
                    possible_indices.remove(best_path_idx)
                    self.multi_tracker.global_batches_tracks[best_path_idx].extend(path)
                    placed = True
                    
                    if placed == False:
                        raise ValueError
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            batch_count += 1
            if batch_count == 10: 
                a=False

        self.drawer.plot_batch_paths(self.multi_tracker.global_batches_tracks, config.BATCH_PLOTS_PATH)
        # self.multi_tracker.save_tracking_history(self.tracking_history_path)
        # self.drawer.plot_logs(self.tracking_history_path, self.output_plot_path)

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()



