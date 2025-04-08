import cv2
import numpy as np
from .BaseDetector import BaseDetector

class ColorSegmentationDetector(BaseDetector):

    def __init__(self, lower_color_range, upper_color_range, contour_area_threshold):
        self.lower_color_range = np.array(lower_color_range)     
        self.upper_color_range = np.array(upper_color_range) 
        self.contour_area_threshold = contour_area_threshold  

    def detect(self, frame, print_detected=True):
        # If testing with image (jpg, jpeg, png), read the image
        #frame = cv2.imread(frame)

        # Check if the image is loaded correctly
        if frame is None:
            raise ValueError("Image not loaded correctly. Please check the file path.")

         # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Range for pink/brown colors 
        lower_pink = self.lower_color_range
        upper_pink = self.upper_color_range
        
        # Create mask
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # filter small contours
            if cv2.contourArea(contour) > self.contour_area_threshold:  
                x1, y1, w, h = cv2.boundingRect(contour)
                x2 = x1 + w 
                y2 = y1 + h 
                cx = x1 + (w // 2)
                cy = y1 + (h // 2)
                
                out = {'bbox_xyxy': (x1, y1, x2, y2), 'center': (cx, cy),'conf': 0.3, 'cls': 300}
                
                detections.append(out)
        
        if print_detected:
            print(f"Detected {len(detections)} pigs.")
            print(f"Detected pigs: {detections}")

        return detections