import src.modules as modules
import src.config as config
import cv2
import os

class PigMonitor:
    
    def __init__(self):
        self.detector = modules.Detector()
        self.mapper = modules.Mapper(config.MAPPINGS)
        self.multi_tracker = modules.MultiTracker(config.NUM_CAMERAS, config.FIRST_CAMERA, self.mapper, config.CLUSTER_EPSILON, config.CLUSTER_MIN_SAMPLES, config.MAX_GLOBAL_AGE, config.MAX_CLUSTER_DISTANCE, False)
        self.drawer = modules.Drawer()
        self.file_directory = config.MEDIAFLUX_VIDEO_DIR
        self.sync = modules.Synchronizer(self.file_directory)
        self.video_writer_path = config.PROCESSED_VIDEO_PATH
        self.tracking_history_path = config.TRACKING_HISTORY_PATH
        self.first_camera = 5
        self.frame_number = 0
        self.max_frames = 1000

    def process_frame(self, frame, frame_count, cam_id=None):

        # Detect pigs in the frame
        detections = self.detector.detect(frame)

        # Update tracks 
        tracks = self.multi_tracker.track(detections, cam_id)

        # Draw tracks on the frame
        display_frame = self.drawer.draw_bboxes(frame.copy(), tracks)
        display_frame = self.drawer.add_useful_info(display_frame, frame_count, tracks)

        return display_frame, tracks

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

    def multi_monitor(self):
        """Processes all video files in the directory specified in config."""
        # Check if directory exists
        if not os.path.exists(self.file_directory):
            print(f"Error: Directory {self.file_directory} not found.")
            return
        
        # Get all video files in the directory
        files = os.listdir(self.file_directory)
        video_files = [f for f in files if f.endswith(('.mp4', '.avi'))]

        if not video_files:
            print("No video files found in the directory.")
            return

        video_caps = {}
        fps_dict = {}

        # Synchronize videos
        sorted_videos = self.sync.separate_by_cameras(video_files)
        camera_offsets = self.sync.get_offsets()
        video_caps, fps_dict = self.sync.synchronize(sorted_videos, camera_offsets)
        
        print("VIDEO CAPTURE OBJECTS INITIALIZED")
        print(video_caps)

        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Get video properties
        width = config.OUTPUT_VIDEO_WIDTH
        height = config.OUTPUT_VIDEO_HEIGHT
        fps = config.OUTPUT_VIDEO_FPS
        out = cv2.VideoWriter(self.video_writer_path, fourcc, fps, (width, height))

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

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()

