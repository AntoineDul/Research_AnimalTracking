import cv2

# This is a helper class to extract frames from a video file in order to analyze them separately. 
# It was origianly created to get the frames from each camera to find the pixels to match with real world coordinates to compute the homography matrix.

def get_screenshot(video_path, cam_id, frame_number, offset_from_base):
    """
    Extract a single frame from a video file and save it as an image.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the extracted frame.
        frame_number (int): Frame number to extract.
    """
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_frame = frame_number
    if target_frame >= total_frames:
        print(f"Warning: Frame {target_frame} exceeds total frame count {total_frames} in video {video_path}")
        cap.release()
        return
    
    # Read the frame
    ret, frame = cap.read()
    
    output_path = f"C:\\Users\\antoi\\OneDrive\\Documents\\UniMelb\\COMP30013\\PigMonitor\\outputs\\screenshots\\camera_{cam_id}_offset_{offset_from_base}.jpg"

    if ret:
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Saved screenshot to {output_path}")
    else:
        print("Error: Could not read the frame.")
    
    # Release the video capture object
    cap.release()

if __name__ == "__main__":

    video_path_D5 = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08\\D5_S20241016075913_E20241016080735.mp4"
    video_path_D6 = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08\\D6_S20241016080005_E20241016080827.mp4"
    video_path_D7 = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08\\D7_S20241016080133_E20241016080955.mp4"
    video_path_D8 = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08\\D8_S20241016080024_E20241016080846.mp4"
    video_path_D17 = "C:\\Users\\antoi\\Documents\\Unimelb_Mediaflux\\2024-10-16~08\\D17_S20241016080531_E20241016081353.mp4"
    video_paths = [video_path_D5, video_path_D6, video_path_D7, video_path_D8, video_path_D17]

    # frame_numbers = [1200, 1040, 0, 1200, 1400]  
    base_frame_numbers = [2800, 1760, 0, 1380, 5280]
    offsets = [int(x * 20) for x in [0, 25, 40, 72, 107.5]]
    cam_ids = [5, 6, 7, 8, 17]
    for offset in offsets: 
        for i in range(len(video_paths)):
            get_screenshot(video_paths[i], cam_ids[i], base_frame_numbers[i] + offset, offset)
        

# start of common videos 
# FRAME OFFSET CAM 9: 5280
# FRAME OFFSET CAM 5: 2800
# FRAME OFFSET CAM 6: 1760
# FRAME OFFSET CAM 7: 0
# FRAME OFFSET CAM 8: 1380