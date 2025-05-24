import numpy as np
import cv2
import pickle

class Mapper:

    def __init__(self, mappings, resolution, distortion):
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

    def image_to_world_coordinates(self, cam_id, image_points):
        if cam_id not in self.cameras:
            raise ValueError(f"Camera ID {cam_id} not found in mappings: {self.cameras}.")

        homography_matrix = self.homography_matrices[cam_id % self.cameras[0]]
        world_points = cv2.perspectiveTransform(image_points, homography_matrix)[0][0]
        # print("World Points: ", world_points)
        return world_points

    def batch_to_world_coords(self, paths_list, cam_id): # paths_list = [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ]
        transformed_paths = []
        for path in paths_list:
            transformed_path = []
            for i in range(len(path)):
                data = path[i]
                cam_coords = np.array([[[data[0][0], data[0][1]]]], dtype=np.float32) # Shape (1, 1, 2)
                global_coords = self.image_to_world_coordinates(cam_id, cam_coords)
                global_data = ((float(global_coords[0]), float(global_coords[1])), data[1])
                transformed_path.append(global_data)
            transformed_paths.append(transformed_path)
        return transformed_paths

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
    
    def load_model(self, cam_id, models_dir):
        path = f"{models_dir}/cam_{cam_id}_bias_model.pkl"
        with open(path, "rb") as f:
            models = pickle.load(f)
            model_dx = models["dx"]
            model_dy = models["dy"]
        return model_dx, model_dy

    @staticmethod
    def correct_bias(input_position:tuple, cam_id:int, scales:dict, camera_positions:dict):

        # Camera position
        cam_pos = camera_positions[cam_id]

        # Convert relative position to meters
        rel_x = (input_position[0] - cam_pos[0])
        rel_y = (input_position[1] - cam_pos[1])
        # print(f"point relative to position camera: ({rel_x}, {rel_y})")

        # Apply Thales scaling 
        cam_scale_x, cam_scale_y = scales[cam_id]
        
        # If pigs are close to cam position, no need to unbias
        if abs(rel_x) < 0.4 and abs(rel_y) < 0.4:
            return input_position

        elif abs(rel_x) < 0.4 and abs(rel_y)>= 0.4:
            corrected_rel_y = rel_y * cam_scale_y
            unbiased_y = cam_pos[1] + corrected_rel_y
            return (input_position[0], unbiased_y)
        
        elif abs(rel_x) >= 0.4 and abs(rel_y) < 0.4:
            corrected_rel_x = rel_x * cam_scale_x
            unbiased_x = cam_pos[0] + corrected_rel_x
            return (unbiased_x, input_position[1])
        
        else:

            corrected_rel_x = rel_x * cam_scale_x
            corrected_rel_y = rel_y * cam_scale_y

            # Final unbiased position in tiles
            unbiased_x = cam_pos[0] + corrected_rel_x
            unbiased_y = cam_pos[1] + corrected_rel_y

            return (unbiased_x, unbiased_y)

    def fix_paths_bias(self, paths:dict, scales:dict, cam_positions:dict):
        unbiased_paths = {}

        for cam_id, batch in paths.items():
            new_batch = []
            
            # translate to real world coord
            world_coord_batch = self.batch_to_world_coords(batch, cam_id) 

            # apply bias correction on each point of each path
            for path in world_coord_batch:
                if len(path) == 0:
                    continue
                new_path = []
                for point in path:
                    if len(point) == 0: 
                        continue
                    # print(point)
                    unbiased_point = self.correct_bias(point[0], cam_id, scales, cam_positions)
                    # print(f"unbiased point : {unbiased_point}")
                    assert(isinstance(point, tuple))

                    new_path.append((unbiased_point, point[1]))
                new_batch.append(new_path)
            unbiased_paths[cam_id] = new_batch

        return unbiased_paths