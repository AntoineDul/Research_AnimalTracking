import numpy as np
import cv2

class Mapper:

    def __init__(self, mappings, resolution, distortion):
        self.w, self.h = resolution
        self.k = np.array([
                [self.w, 0, self.w / 2],
                [0, self.w, self.h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        self.d = np.array(distortion, dtype=np.float64)        # [k1, k2, p1, p2] where k1, k2 -> radial distortion and p1, p2 -> tangential distortion
        self.cameras = list(int(cam_id) for cam_id in mappings.keys())
        self.homography_matrices = [
            cv2.findHomography(mapping['image_points'], mapping['world_points'])[0] for mapping in mappings.values()
        ]

    def image_to_world_coordinates(self, cam_id, image_points):
        """Translate a single point from cam relative coordinates to pen coordinates"""
        if cam_id not in self.cameras:
            raise ValueError(f"Camera ID {cam_id} not found in mappings: {self.cameras}.")

        homography_matrix = self.homography_matrices[cam_id % self.cameras[0]]
        world_points = cv2.perspectiveTransform(image_points, homography_matrix)[0][0]
        return world_points

    def batch_to_world_coords(self, paths_list, cam_id): 
        """Translate every point from a path from cam relative coordinates to pen coordinates
        paths_list format: [ [((x1, y1), f1), ((x2, y2), f2), ...], ... ] where f is frame number
        """
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
    
    def undistort_images(self, image):
        """Remove the fisheye effect from the input frame (image)"""
        try:
            image.shape
        except AttributeError:
            image = cv2.imread(image)

        return cv2.fisheye.undistortImage(image, self.k, self.d, Knew=self.k)

    @staticmethod
    def correct_bias(input_position:tuple, cam_id:int, scales:dict, camera_positions:dict):
        """Correct the bias of a single input point"""

        # Camera position
        cam_pos = camera_positions[cam_id]

        # Convert relative position to meters
        rel_x = (input_position[0] - cam_pos[0])
        rel_y = (input_position[1] - cam_pos[1])

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
        """Correct the bias for each points of the dictionary of lists of paths input"""
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
                    unbiased_point = self.correct_bias(input_position=point[0], cam_id=cam_id, scales=scales, camera_positions=cam_positions)
                    assert(isinstance(point, tuple))

                    new_path.append((unbiased_point, point[1]))
                new_batch.append(new_path)
            unbiased_paths[cam_id] = new_batch

        return unbiased_paths