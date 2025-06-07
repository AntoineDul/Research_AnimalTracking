import csv
import json
import math
import config
from collections import defaultdict
import re
import matplotlib.pyplot as plt

DISTANCE_THRESHOLD = 0.2

def load_ground_truth(csv_path):
    gt_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["Frame difference"])
            gt_data[frame] = {}
            for i in range(1, 16):
                key = f"Pig ID {i}"
                val = row[key].strip('"')
                if val.lower() == "box" or not val:
                    continue
                match = re.match(r"\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?", val)
                if match:
                    coords = (float(match.group(1)), float(match.group(2)))
                    gt_data[frame][i] = coords
                
    return gt_data

def load_tracked(json_path, check_frames):
    with open(json_path, 'r') as f:
        data = json.load(f)

    tracked = {}
    for pig_id, path in enumerate(data):
        # path is a list of [[x,y], frame] pairs
        tracked[pig_id] = [(tuple(point), frame) for point, frame in path if frame in check_frames]
    return tracked

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_track_lengths(tracked):
    return [len(track_data) for track_data in tracked.values()]

def plot_track_length_distribution(track_lengths, output_file='track_lengths.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(track_lengths, bins=20, color='steelblue', edgecolor='black')
    plt.title('Distribution of Track Lengths')
    plt.xlabel('Track Length (number of detections)')
    plt.ylabel('Number of Tracks')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def compute_average_successful_track_length(tracked, threshold=5):
    lengths = [len(t) for t in tracked.values() if len(t) >= threshold]
    return sum(lengths) / len(lengths) if lengths else 0

def count_tracked_pigs_per_timestamp(tracked, gt_timestamps):
    timestamp_counts = defaultdict(int)
    for track_id, positions in tracked.items():
        for pos, ts in positions:
            if ts in gt_timestamps:
                timestamp_counts[ts] += 1
    return timestamp_counts

def plot_tracked_pigs_over_time(timestamp_counts, output_file='tracked_pigs_over_time.png'):
    timestamps = sorted(timestamp_counts.keys())
    
    counts = [timestamp_counts[ts] for ts in timestamps]
    timestamps = [timestamp / 1200 for timestamp in timestamps]
    plt.figure(figsize=(12, 12))
    plt.plot(timestamps, counts, marker='o', color='darkgreen')
    plt.title('Total Number of Tracked Pigs per Minute', fontsize=28)
    plt.xlabel('Time (minutes)', fontsize=24)
    plt.ylabel('Number of Tracked Pigs', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 14)  # Set y-axis from 0 to 14
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

    print(f"Saved pig count over time plot as {output_file}")

def evaluate(tracked_json, ground_truth_csv):
    gt = load_ground_truth(ground_truth_csv)
    check_frames = gt.keys()
    print("gt: ",gt)
    tracked = load_tracked(tracked_json, check_frames)
    print("tracked: ", tracked)

    # 1. Initial matching at first timestamp
    first_ts = min(gt.keys())
    gt_first = gt[first_ts]
    track_first = {track_id: det[0][0] for track_id, det in tracked.items() if det and det[0][1] == first_ts}

    gt_to_track = {}
    used_tracks = set()

    for gt_id, gt_pos in gt_first.items():
        min_dist = float('inf')
        matched_track_id = None
        for track_id, track_pos in track_first.items():
            if track_id in used_tracks:
                continue
            dist = math.dist(gt_pos, track_pos)
            if dist < min_dist:
                min_dist = dist
                matched_track_id = track_id
        if matched_track_id is not None:
            gt_to_track[gt_id] = matched_track_id
            used_tracks.add(matched_track_id)

    # 2. Tracking duration evaluation
    track_success = defaultdict(int)

    # Go through each timestamp in order
    for ts in sorted(gt.keys()):
        gt_frame = gt[ts]
        for gt_id, track_id in gt_to_track.items():
            # Check if tracking data exists at this timestamp
            if track_id not in tracked:
                continue
            found = False
            for track_pos, track_ts in tracked[track_id]:
                if track_ts == ts:
                    if gt_id in gt_frame:
                        dist = math.dist(gt_frame[gt_id], track_pos)
                        if dist < DISTANCE_THRESHOLD:
                            track_success[gt_id] += 1
                    found = True
                    break
            # If no position at this timestamp, skip
            if not found:
                continue

    # 3. Output results
    for gt_id in sorted(gt_to_track):
        print(f"Pig {gt_id} (matched to track ID {gt_to_track[gt_id]}): Tracked successfully for {track_success[gt_id]} minutes.")

    # 4. Compute and plot track length statistics
    track_lengths = compute_track_lengths(tracked)
    avg_length = compute_average_successful_track_length(tracked, threshold=5)
    print(f"\nAverage length of successful tracks (length ≥ 5): {avg_length:.2f} detections")

    plot_track_length_distribution(track_lengths)
    
    # 5. Plot number of tracked pigs per timestamp (per minute)
    timestamp_counts = count_tracked_pigs_per_timestamp(tracked, gt.keys())
    plot_tracked_pigs_over_time(timestamp_counts)


if __name__ == "__main__":
    # Input the path to the json file containing the tracking history and the path to the csv file containing the ground truth
    evaluate(tracked_json=f"{config.TRACKING_HISTORY_PATH}\\20250604_144202_track_200_batch-size_100_batches.json", ground_truth_csv=config.CSV_PATH)