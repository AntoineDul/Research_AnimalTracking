# PigMonitor: Multi-Camera Pig Tracking Pipeline

## Overview

The PigMonitor package is a multi-camera video analysis pipeline designed to detect and track the movement of pigs in a pen using synchronized video feeds. The core of the system is the `batch_monitor` function, which processes video data in batches, merges detections across multiple camera views, and outputs tracking results and visualizations.

### Processing Pipeline (`batch_monitor`)

1. **Setup & Synchronization**  
   - The system loads and synchronizes video files from multiple cameras using timestamps and camera IDs.
   - Video capture objects are initialized for each camera.

2. **Batch Processing**  
   - Video frames are processed in batches (e.g., 200 frames per batch). 
   - For each frame in the batch and each camera:
     - Frames are undistorted and mapped to a common coordinate system.
     - Pigs are detected using a YOLO-based detector.
     - Detected pigs are tracked within each camera view.

3. **Path Correction & Merging**  
   - Detected paths are corrected for camera bias and outliers.
   - Paths are merged across cameras to reconstruct global pig trajectories.
   - Duplicate and incomplete paths are handled to improve tracking accuracy.

4. **Logging & Visualization**  
   - Tracking results and logs are saved to output files.
   - Batch and global path plots visualizations are generated for analysis. Only the first 10 batches generate plot as they get messy for later batches.

5. **Output**  
   - Tracking histories are saved as JSON files.
   - Plots are saved to the output directory.

## How to Run

### 1. Install Dependencies

Ensure you have Python 3.12.7 and all required packages installed.

```bash
pip install -r requirements.txt
```

### 2. Configure `config.py`

Edit `src/config.py` to set up your environment:

- **Directories & Paths**  
  - `MEDIAFLUX_VIDEO_DIR`: Path to all video files to process. They must have the following naming format: D{cam_id}_S{year}{month}{day}{hours}{minutes}{seconds}_E{year}{month}{day}{hours}{minutes}{seconds}, where S represent the start time and E the end time of the footage. There must be no gap between two videos of the same camera. 
  - `OUTPUT_DIR`: Where outputs (videos, logs, plots) will be saved. The directory must exist before running the program.
  - `YOLO_MODEL_PATH`: Path to the YOLO model weights.
  - `TRACKING_HISTORY_PATH`, `BATCH_PLOTS_PATH`, etc.: Output subdirectories.

- **Farm & Camera Settings**  
  - `NUM_PIGS`: Number of pigs present in the pen.
  - `NUM_CAMERAS`: The number of cameras recording the pen.
  - `FIRST_CAMERA`: Lowest camera ID.
  - `CAM_ID_TO_CHANGE`: Camera IDs must be sequential, so if the ids are not, configure this dictionary to correct the camera IDs needed.
  - `RESOLUTION`, `DISTORTION`: Camera calibration parameters.
  - `CAM_POSITIONS`, `MAPPINGS`: Camera positions and homography mappings.

- **Detection & Tracking Parameters**: adjust these if the pipeline is not producing accurate tracking.  
  - `YOLO_CONF_THRESHOLD`: Detection confidence threshold.
  - `BATCH_SIZE`: Number of frames per batch.
  - `FRECHET_THRESHOLD`, `SIMILARITY_THRESHOLD`: Path matching parameters.

- **Output Settings**  
  - `OUTPUT_VIDEO_WIDTH`, `OUTPUT_VIDEO_HEIGHT`, `OUTPUT_VIDEO_FPS`: Output video properties.

### 3. Run the Pipeline

From the `PigMonitor` directory, run the main script:

```bash
python -m src.PigMonitor
```

Or, if you have a main entry point, adjust accordingly.

## Notes

- Ensure all paths in `config.py` are correct and accessible.
- The pipeline expects synchronized video files named according to camera IDs.
- Output files (videos, logs, JSONs, plots) will be saved in the directories specified in `config.py`.

## Troubleshooting

- If you encounter import errors, check your Python path and run scripts from the `PigMonitor` root directory.
- For camera calibration and mapping, ensure the `MAPPINGS` dictionary in `config.py` is filled with accurate image and world points for your setup.

---

For further details, refer to the code comments and docstrings in the source files.
