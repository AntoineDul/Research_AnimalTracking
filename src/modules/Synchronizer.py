import pandas as pd
from datetime import datetime

class Synchronizer:
    def __init__(self, rfid_data_path=None):
        # self.rfid = pd.read_excel(rfid_data_path)
        self.sorted_video_times = {}
        self.offsets = {}
        self.TIMES = 0
        self.FILENAME = 1
        self.START = 0
        self.END = 1
        self.frame_source1 = None
        self.frame_source2 = None
        self.frame_source3 = None
        self.frame_source4 = None

    def get_relevant_rfid(self, start_time, end_time):
        # Filter RFID data based on the time range
        pass

    def synchronize(self):
        # Implement the synchronization logic here
        pass
    
    @staticmethod
    def parse_filename(filename):
        # D{camera_number}_S{start_timestamp}_E{end_timestamp}
        parts = filename.split('_')
        cam_id = int(parts[0][1:])  # 'D1' -> 1
        start_str = parts[1][1:]  # 'S20240408093000' -> '20240408093000'
        end_str = parts[2][1:].split('.')[0]  # 'E20240408100000' -> '20240408100000'
        start_time = datetime.strptime(start_str, '%Y%m%d%H%M%S')
        end_time = datetime.strptime(end_str, '%Y%m%d%H%M%S')
        return cam_id, start_time, end_time
    
    def separate_by_cameras(self, video_files):
        # Separate videos by camera ID
        videos = {}
        file_info = [(self.parse_filename(f), f) for f in video_files]
        for ((cam_id, start_time, end_time), file) in file_info:
            if cam_id not in videos:
                videos[cam_id] = []
            videos[cam_id].append(((start_time, end_time), file))

        self.sorted_video_times = {k: sorted(v, key=lambda x: x[0][self.START]) for k, v in videos.items()}
        return self.sorted_video_times  # {cam_id: [((start, end), filename), ...]}
        

    def get_offsets(self):
        # Get offsets for each camera wrt to the first camera that started
        if not self.sorted_video_times:
            raise ValueError("No sorted videos to calculate offsets from.")
    
        global_start = max([self.sorted_video_times[cam_id][0][self.TIMES][self.START] for cam_id in self.sorted_video_times.keys()])
        for cam_id, times_videos in self.sorted_video_times.items():
            start_time = times_videos[0][self.TIMES][self.START]
            offset = (global_start - start_time).total_seconds()
            self.offsets[cam_id] = offset
        return self.offsets
