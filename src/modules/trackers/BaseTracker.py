class BaseTracker:

    def track(self, detections):
        """
        Abstract method to track objects in the current frame based on previous detections.
        :param tracked: List of tracked objects from the previous frame.
        :param detections: List of detected objects in the current frame.
        :return: Updated list of tracked objects.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def get_tracks(self):
        """
        Abstract method to get the current tracks.
        :return: List of currently tracked objects.
        """
        
        raise NotImplementedError("Subclasses must implement this method.")