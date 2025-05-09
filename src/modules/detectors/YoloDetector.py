from ultralytics import YOLO
from .BaseDetector import BaseDetector
import cv2

class YoloDetector(BaseDetector):
    def __init__(self, model_path, conf_threshold, yolo_imgsz):
        self.model = YOLO(model_path)  # Load the YOLO model
        self.conf_threshold = conf_threshold
        self.yolo_imgsz = yolo_imgsz

    def resize_output(self, original_h, original_w, x1, y1, x2, y2):
        """Resize the bounding box coordinates back to the original image size."""
        x_1 = int(x1 * original_w / self.yolo_imgsz)
        y_1 = int(y1 * original_h / self.yolo_imgsz)
        x_2 = int(x2 * original_w / self.yolo_imgsz)
        y_2 = int(y2 * original_h / self.yolo_imgsz)
        cx = abs((x_1 + x_2) // 2)
        cy = abs((y_1 + y_2) // 2)

        return x_1, y_1, x_2, y_2, cx, cy

    def detect(self, frame, print_detected=False):
        # TODO: Add image enhancement before processing them
        
        try :
            original_h, original_w = frame.shape[:2]  # Get original size
        except AttributeError:
            # If testing with image (jpg, jpeg, png), read the image
            frame = cv2.imread(frame)
            original_h, original_w = frame.shape[:2]

        resized_frame = cv2.resize(frame, (self.yolo_imgsz, self.yolo_imgsz))  # Resize to expected yolo input size

        pigs_detected = []

        results = self.model(resized_frame)  # Run YOLO
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box bbox_xyxy
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID
                

                if conf >= self.conf_threshold:
                    # Scale bbox back to original frame size
                    x1, y1, x2, y2, cx, cy = self.resize_output(original_h, original_w, x1, y1, x2, y2)

                    out = {'bbox_xyxy': (x1, y1, x2, y2), 'center':(cx, cy), 'conf': conf.item(), 'cls': cls}
                    pigs_detected.append(out)

        if print_detected:
            print(f"Detected {len(pigs_detected)} pigs.")
            print(f"Detected pigs: {pigs_detected}")

        return pigs_detected