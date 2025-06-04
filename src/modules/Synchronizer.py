from datetime import datetime
import cv2
import os

class Synchronizer:
    def __init__(self, cam_id_to_change, file_directory=None, rfid_data_path=None):
        # self.rfid = pd.read_excel(rfid_data_path)
        self.file_directory = file_directory
        self.sorted_video_times = {}
        self.offsets = {}
        self.TIMES = 0
        self.FILENAME = 1
        self.START = 0
        self.END = 1
        self.cam_id_to_change = cam_id_to_change

    # def get_relevant_rfid(self, start_time, end_time):
    #     # Filter RFID data based on the time range
    #     pass

    def synchronize(self):
        video_caps = {}
        fps_dict = {}

        # Iterate through each camera and set the video capture object
        for cam_id, time_files in self.sorted_video_times.items():   
            file_data = time_files.pop(0)     
            first_file_name = file_data[1]             # Get the first video file for each camera
            full_path = os.path.join(self.file_directory, first_file_name)
            cap = cv2.VideoCapture(full_path)
            fps = cap.get(cv2.CAP_PROP_FPS)     # 20 fps for all farm videos
            offset_sec = self.offsets[cam_id]
            offset_frames = int(offset_sec * fps)
            print(f"FPS {cam_id}:", fps)
            print(f"FRAME OFFSET CAM {cam_id}:", offset_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset_frames)
            video_caps[cam_id] = cap
            fps_dict[cam_id] = fps

        return video_caps, fps_dict
    
    def parse_filename(self, filename):
        """Extract  cam_id, start time and end time from a given file name.
        file format expected: D{camera_number}_S{start_timestamp}_E{end_timestamp}
        """
    
        parts = filename.split('_')
        cam_id = int(parts[0][1:])  # Remove 'D' from start of file name e.g. ('D1' -> 1)

        # Replace desired cam_ids with new ones 
        if cam_id in self.cam_id_to_change.keys():
            cam_id = self.cam_id_to_change[cam_id]

        start_str = parts[1][1:]  # Remove 'S' from before the starting time (e.g. 'S20240408093000' -> '20240408093000')
        end_str = parts[2][1:].split('.')[0]  # Remove 'E' from end time: (e.g. 'E20240408100000' -> '20240408100000')
        start_time = datetime.strptime(start_str, '%Y%m%d%H%M%S')
        end_time = datetime.strptime(end_str, '%Y%m%d%H%M%S')
        return cam_id, start_time, end_time
    
    def separate_by_cameras(self, video_files):
        """Separate videos by camera ID"""

        videos = {}
        
        file_info = [(self.parse_filename(f), f) for f in video_files]
        for ((cam_id, start_time, end_time), file) in file_info:
            if cam_id not in videos:
                videos[cam_id] = []
            videos[cam_id].append(((start_time, end_time), file))
        
        self.sorted_video_times = {k: sorted(v, key=lambda x: x[0][self.START]) for k, v in videos.items()}
        
        # Check there is no time gap between two recording from same pov
        for cam_id, all_video_data in self.sorted_video_times.items():
            for i in range(len(all_video_data)):
                if i == 0: continue
                try:
                    assert all_video_data[i - 1][0][self.END] == all_video_data[i][0][self.START]
                except AssertionError:
                    raise ValueError(f"Videos must not have time gaps in order to be processed ({cam_id}).")
        return self.sorted_video_times.copy()  # {cam_id: [((start, end), filename), ...]}

    def get_offsets(self):
        """Get offsets for each camera w.r.t. to the first camera that started"""
        if not self.sorted_video_times:
            raise ValueError("No sorted videos to calculate offsets from.")
    
        global_start = max([self.sorted_video_times[cam_id][0][self.TIMES][self.START] for cam_id in self.sorted_video_times.keys()])
        for cam_id, times_videos in self.sorted_video_times.items():
            start_time = times_videos[0][self.TIMES][self.START]
            offset = (global_start - start_time).total_seconds()
            self.offsets[cam_id] = offset
        return self.offsets
    
    def start_next_video(self, cam_id):
        time_files = self.sorted_video_times[cam_id]

        # Check if there are some videos left 
        if len(time_files) == 0: 
            return None
        
        file_data = time_files.pop(0)     
        file_name = file_data[1]             # Get the first video file for each camera
        full_path = os.path.join(self.file_directory, file_name)
        cap = cv2.VideoCapture(full_path)

        return cap      

