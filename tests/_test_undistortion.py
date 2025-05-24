import cv2
import os 
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.modules.Mapper import Mapper
import src.config as config

# k1_values = [-0.4, -0.3, -0.2, -0.1, 0.0]
# k2_values = [0.0, 0.05, 0.1]

k1_values = [-0.6]
k2_values = [0.1]


def undistort(cam_id):
    for k1 in k1_values:
        for k2 in k2_values:
            mapper = Mapper(mappings=config.MAPPINGS, resolution=config.RESOLUTION, distortion=[k1,k2,0,0])
            undistorted_image = mapper.undistort_images(f"C:\\Users\\antoi\\OneDrive\\Documents\\UniMelb\\COMP30013\\PigMonitor\\outputs\\screenshots\\camera_{cam_id}.jpg")

            if undistorted_image is None:
                print("fail")

            cv2.imwrite(f"outputs/undistorted_cameras/camera_{cam_id}_k1_{k1}_k2_{k2}.png", undistorted_image)
            print("saved")


def undistort_all_screenshots():
    mapper = Mapper(mappings=config.MAPPINGS, resolution=config.RESOLUTION, distortion=config.DISTORTION)
    directory = "C:\\Users\\antoi\\OneDrive\\Documents\\UniMelb\\COMP30013\\PigMonitor\\outputs\\screenshots"
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            full_file = f"C:\\Users\\antoi\\OneDrive\\Documents\\UniMelb\\COMP30013\\PigMonitor\\outputs\\screenshots\\{file}"
            undistorted_image = mapper.undistort_images(full_file)
            cv2.imwrite(f"outputs/undistorted_cameras/undistorted_{file}", undistorted_image)

if __name__ == "__main__":
    undistort_all_screenshots()