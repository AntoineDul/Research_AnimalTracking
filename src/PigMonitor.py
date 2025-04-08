import src.modules as modules
import src.config as config
import cv2
import os

class PigMonitor:
    
    def __init__(self):
        self.detector = modules.Detector()
        self.tracker = modules.Tracker()
        self.drawer = modules.Drawer()

    def process_frame(self, frame):

        # Detect pigs in the frame
        detections = self.detector.detect(frame)

        # Update tracks 
        tracks = self.tracker.track(detections)

        # Draw tracks on the frame
        display_frame = self.drawer.draw_bboxes(frame.copy(), tracks)

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
        print("Press 'c' to toggle color-based detection only.")

        frame_count = 0
        screenshot_count = 0
        color_detection_only = False
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to improve performance (process every 2nd frame)
            if frame_count % config.FRAME_SKIP != 0 and frame_count > 1:
                continue
                
            # Process frame
            processed_frame, tracks = self.process_frame(frame)
            
            # Add frame number and threshold info
            cv2.putText(
                processed_frame,
                f"Frame: {frame_count} | Det Conf Threshold: {config.YOLO_CONF_THRESHOLD:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            
            # Add pig count
            cv2.putText(
                processed_frame,
                f"Detections: {len(tracks)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
            
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