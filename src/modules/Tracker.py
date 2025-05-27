from src import config
from .trackers.GreedyTracker import GreedyTracker
from .trackers.DeepSortTracker import DeepSortTracker

class Tracker:
        
    def __init__(self):
        self.tracker = self.load_tracker()

    def load_tracker(self):
        if config.TRACKING_METHOD == "GREEDY":
            return GreedyTracker(config.IOU_THRESHOLD, config.MAX_TRACK_AGE, config.MAX_DISTANCE, config.ALPHA)
        elif config.TRACKING_METHOD == "DEEP_SORT":
            return DeepSortTracker()
        else:
            raise ValueError(f"Unknown model: {config.DETECTION_METHOD}")

    def track(self, frame):
        
        """
        Tracking function

        Args:
            frame: The current video frame being processed.

        Returns:
            list: List of currently tracked objects. Each object is represented as a dictionary with keys 'id', 'bbox_xyxy', 'center', 'cls', 'conf' and 'age'.
        """

        return self.tracker.track(frame)

    def get_tracks(self):
        
        """
        Get the current tracks.

        Returns:
            list: List of currently tracked objects. Each object is represented as a dictionary with keys 'id', 'bbox_xyxy', 'center', 'cls', 'conf' and 'age'.
        """

        return self.tracker.get_tracks()

    def reinitialize_id_count(self):
        self.tracker.reinitialize_id_count()