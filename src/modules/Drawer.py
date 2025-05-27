import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Drawer:

    @staticmethod
    def draw_bboxes(frame, tracks):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (numpy.ndarray): The image frame to draw on.
            tracks (list): List of tracked objects with their properties.

        Returns:
            numpy.ndarray: The frame with drawn bounding boxes and labels.
        
        """

        for track in tracks:
            x1, y1, x2, y2 = track["bbox_xyxy"]
            
            # Select color based on class
            if track["cls"] == 0:
                color = (0, 255, 0)  # Green for confirmed pigs
            else:
                color = (255, 0, 0)  # Red for other classes
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display ID and class
            label = f"#{track['id']} {track['cls']} {track['conf']:.2f}"
            cv2.putText(
                frame, 
                label,
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

        return frame
    
    @staticmethod
    def add_useful_info(processed_frame, frame_count, tracks):
        # Add frame number and threshold info
        cv2.putText(
            processed_frame,
            f"Frame: {frame_count} | Detections: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        return processed_frame

    @staticmethod
    def make_grid(frames_dict, output_size=(1600, 900)):
         
        # Ensure consistent order (camera 1 to 4)
        frames = [frames_dict[cam_id] for cam_id in sorted(frames_dict.keys())]

        # Resize all frames to the same size (match first one)
        height, width = frames[0].shape[:2]
        resized = [cv2.resize(f, (width, height)) for f in frames]

        # Add cam_id  
        for i, f in enumerate(resized):
            cv2.putText(f, f'Camera {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # Build grid
        top = np.hstack((resized[0], resized[1]))
        bottom = np.hstack((resized[2], resized[3]))
        grid = np.vstack((top, bottom))

        # Resize entire grid to desired output size
        if output_size is not None:
            grid = cv2.resize(grid, output_size)

        return grid
    
    @staticmethod
    def plot_logs(logs_path, plot_path):
        with open(logs_path) as f:
            logs = json.load(f)

        tracks_by_id = {}

        for entry in logs:
            for track in entry["global tracks"]:
                track_id = track["id"]
                x, y = track["center"]
                if track_id not in tracks_by_id:
                    tracks_by_id[track_id] = {"x": [], "y": [], "frames": []}
                tracks_by_id[track_id]["x"].append(x)
                tracks_by_id[track_id]["y"].append(y)
                tracks_by_id[track_id]["frames"].append(entry["frame"])

        # Plot
        plt.figure(figsize=(8, 6))
        for track_id, track in tracks_by_id.items():
            print(f"Pig {track_id}: {len(track['x'])} positions")
            plt.plot(track["x"], track["y"], label=f"Pig {track_id}")
            plt.scatter(track["x"], track["y"], s=5)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory of Each Pig")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.savefig(plot_path)
        plt.show()

    @staticmethod
    def plot_batch_paths(data, cam_id, batch_size, output_path, batch_count):
        plt.figure(figsize=(8, 6))
        plt.title(f'Cam {cam_id} view - Paths - {batch_size}')
        cmap = cm.get_cmap('tab20', len(data))  # Use a colormap with enough colors

        for i, path in enumerate(data):
            # print(f"path: {path}")
            if len(path) > 0:
                x_coords = [point[0][0] for point in path if len(point) is not None and len(point[0]) > 0]
                y_coords = [point[0][1] for point in path if len(point) is not None and len(point[0]) > 0]
                if len(x_coords) > 0 and len(y_coords) > 0:
                    plt.plot(x_coords, y_coords, color=cmap(i), label=f'Path {i}')
                    

        plt.xlim(0, 2.5)    # set x-axis limits
        plt.ylim(-3, 3)     # set y-axis limits
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        # plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}\\batch_{batch_count}_batch-size_{batch_size}_cam_{cam_id}_batch_plot.jpg")
        print(f"Batch plot for camera {cam_id} saved.")
        # plt.show()

    @staticmethod
    def display_plot_batch(data, cam_id, batch_size):
        plt.figure(figsize=(8, 6))
        plt.title(f'Cam {cam_id} view - Paths - {batch_size}')
        cmap = cm.get_cmap('tab20', len(data))  # Use a colormap with enough colors

        for i, path in enumerate(data):
            # print(f"path: {path}")
            if len(path) > 0:
                x_coords = [point[0][0] for point in path if len(point) is not None and len(point[0]) > 0]
                y_coords = [point[0][1] for point in path if len(point) is not None and len(point[0]) > 0]
                if len(x_coords) > 0 and len(y_coords) > 0:
                    plt.plot(x_coords, y_coords, color=cmap(i), label=f'Path {i}')
                    

        plt.xlim(0, 2.5)    # set x-axis limits
        plt.ylim(-3, 3)     # set y-axis limits
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()
            