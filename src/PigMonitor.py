import src.modules as modules
import src.config as config
import cv2
import os

class PigMonitor:
    
    def __init__(self):
        self.detector = modules.Detector()
        self.trackers = [modules.Tracker(), modules.Tracker(), modules.Tracker(), modules.Tracker()]    # TODO : implement multi tracker to make this cleaner
        self.drawer = modules.Drawer()
        self.sync = modules.Synchronizer()
        self.file_directory = config.MEDIAFLUX_VIDEO_DIR
        self.first_camera = 5

    def process_frame(self, frame, frame_count, cam_id=None):

        # Detect pigs in the frame
        detections = self.detector.detect(frame)

        # Update tracks 
        if cam_id is None:
            tracks = self.trackers[0].track(detections)  # Default to one camera tracker
        else:
            tracks = self.trackers[cam_id % self.first_camera].track(detections)    # Use right tracker for camera

        # Draw tracks on the frame
        display_frame = self.drawer.draw_bboxes(frame.copy(), tracks)
        display_frame = self.drawer.add_useful_info(display_frame, frame_count, tracks)

        return display_frame, tracks

    def monitor(self, video_path):

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
        output_path = "outputs/tracked_pigs.avi"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if not available
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
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
        
        print(f"Tracking complete. Output saved to {output_path}")

    def multi_monitor(self):

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

        sorted_videos = self.sync.separate_by_cameras(video_files)
        camera_offsets = self.sync.get_offsets()

        print(sorted_videos)

        for cam_id, time_files in sorted_videos.items():        
            file = time_files[0][1]             # Get the first video file for each camera
            full_path = os.path.join(self.file_directory, file)
            cap = cv2.VideoCapture(full_path)
            fps = cap.get(cv2.CAP_PROP_FPS)     # 20 fps for all farm videos
            fps = 20
            offset_sec = camera_offsets[cam_id]
            offset_frames = int(offset_sec * fps)
            print("FPS", fps)
            print("FRAME OFFSET", offset_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frames)
            video_caps[cam_id] = cap
            fps_dict[cam_id] = fps
        
        print("VIDEO CAPTURE OBJECTS INITIALIZED")
        print(video_caps)

        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = "outputs/multi_tracked_pigs.avi"

        # Get video properties
        width = config.OUTPUT_VIDEO_WIDTH
        height = config.OUTPUT_VIDEO_HEIGHT
        fps = config.OUTPUT_VIDEO_FPS
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
    
            grid = self.drawer.make_grid(frames)
            out.write(grid)
            cv2.imshow('Synchronized 2x2 Grid', grid)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        for cap in video_caps.values():
            cap.release()
        cv2.destroyAllWindows()

