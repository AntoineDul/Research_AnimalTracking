import numpy as np
import cv2

class Mapper:

    def __init__(self, mappings):
        self.cameras = list(int(cam_id) for cam_id in mappings.keys())
        self.homography_matrices = [
            cv2.findHomography(mapping['image_points'], mapping['world_points'])[0] for mapping in mappings.values()
        ]

    def image_to_world_coordinates(self, cam_id, image_points):
        if cam_id not in self.cameras:
            raise ValueError(f"Camera ID {cam_id} not found in mappings: {self.cameras}.")

        homography_matrix = self.homography_matrices[cam_id % self.cameras[0]]
        world_points = cv2.perspectiveTransform(image_points, homography_matrix)[0][0]
        print("World Points: ", world_points)
        return world_points
    
    def world_to_image_coordinates(self):
        # For backtracking 
        pass
