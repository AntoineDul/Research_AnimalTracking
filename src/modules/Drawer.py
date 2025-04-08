import cv2

class Drawer:
    
    def draw_bboxes(self, frame, tracks):
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