import cv2
import numpy as np

class Drawer:

    @staticmethod
    def draw_bboxes(frame, tracks):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): The image frame to draw on.
            tracks (list): List of tracked objects with their properties.

        Returns:
            numpy.ndarray: The frame with drawn bounding boxes and labels.
        
        """

        for track in tracks:
            x1, y1, x2, y2 = track["bbox_xyxy"]
            
            # Select color based on class
            if track["cls"] == 0:
                color = (0, 255, 0)  # Green for confirmed pigs
            else:
                color = (255, 0, 0)  # Red for other classes
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display ID and class
            label = f"#{track['id']} {track['cls']} {track['conf']:.2f}"
            cv2.putText(
                frame, 
                label,
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

        return frame
    
    @staticmethod
    def add_useful_info(processed_frame, frame_count, tracks):
        # Add frame number and threshold info
        cv2.putText(
            processed_frame,
            f"Frame: {frame_count} | Detections: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        return processed_frame

    @staticmethod
    def make_grid(frames_dict, output_size=(1600, 900)):
         
        # Ensure consistent order (camera 1 to 4)
        frames = [frames_dict[cam_id] for cam_id in sorted(frames_dict.keys())]

        # Resize all frames to the same size (match first one)
        height, width = frames[0].shape[:2]
        resized = [cv2.resize(f, (width, height)) for f in frames]

        # Add cam_id  
        for i, f in enumerate(resized):
            cv2.putText(f, f'Camera {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # Build grid
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        grid = np.vstack((top, bottom))

        # Resize entire grid to desired output size
        if output_size is not None:
            grid = cv2.resize(grid, output_size)

        return grid