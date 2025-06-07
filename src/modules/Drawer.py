import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Drawer:
    
    @staticmethod
    def plot_batch_paths(data, cam_id, batch_size, output_path, batch_count):
        """Plot the paths of the batch just processed of the camera with cam_id input"""
        plt.figure(figsize=(8, 6))
        if cam_id == 'overall':
            plt.title('Overall Merged Paths', fontsize=18)
        else: 
            plt.title(f'Camera {cam_id} Detections - Batch Size {batch_size}', fontsize=18)
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
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        # plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}\\batch_{batch_count}_batch-size_{batch_size}_cam_{cam_id}_batch_plot.jpg")
        print(f"Batch plot for camera {cam_id} saved.")

    @staticmethod
    def display_plot_batch(data, cam_id, batch_size):
        """Debugging function"""
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
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
            