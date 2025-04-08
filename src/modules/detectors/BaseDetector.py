class BaseDetector:
    """Abstract class to ensure unified interface for each detector"""
    
    def detect(self, frame):
        """Method to detect pigs in a frame"""
        raise NotImplementedError("Subclasses must implement detect()")
