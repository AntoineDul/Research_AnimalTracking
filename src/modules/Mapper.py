import numpy as np
import cv2

class Mapper:

    def __init__(self, mappings, resolution, distortion, cam_positions=None):
        self.w, self.h = resolution
        self.k = np.array([
                [self.w, 0, self.w / 2],
                [0, self.w, self.h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        self.d = np.array(distortion, dtype=np.float64) # [k1, k2, p1, p2] where k1, k2 -> adial distortion and p1, p2 -> angential distortion
        self.cameras = list(int(cam_id) for cam_id in mappings.keys())
        self.homography_matrices = [
            cv2.findHomography(mapping['image_points'], mapping['world_points'])[0] for mapping in mappings.values()
        ]

        # Triangulation variables
        # self.rotations = [self.get_rotation_matrix(cam_positions[i]['orientation'][0], cam_positions[i]['orientation'][1], cam_positions[i]['orientation'][2]) for i in cam_positions.keys()]
        # self.translations = [cam_positions[i]['location'] for i in cam_positions.keys()]
        # self.P = [self.k @ np.hstack(r, t) for r, t in zip(self.translations, self.rotations)] # Projection matrices

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
    
    def undistort_images(self, image):
        try:
            image.shape
        except AttributeError:
            image = cv2.imread(image)

        return cv2.fisheye.undistortImage(image, self.k, self.d, Knew=self.k)

    @staticmethod
    def get_rotation_matrix(yaw_deg=0, pitch_deg=-30, roll_deg=0):
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll),  np.cos(roll), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    