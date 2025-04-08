import cv2
import numpy as np
import os

def count_pigs_lab_watershed(frame, show_steps=False):
    original = frame.copy()

    # Step 1: Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Step 2: Threshold A channel (pigs are pink → high A values)
    _, a_mask = cv2.threshold(A, 150, 255, cv2.THRESH_BINARY)

    # Step 3: Optional morphology to clean mask
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(a_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 4: Sure background by dilating
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 5: Sure foreground by distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 6: Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 7: Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 8: Apply watershed
    markers = cv2.watershed(original, markers)
    original[markers == -1] = [0, 0, 255]  # Red boundaries

    # Step 9: Count unique pig regions
    unique_labels = np.unique(markers)
    pig_labels = unique_labels[unique_labels > 1]
    pig_count = len(pig_labels)

    # Visualization
    if show_steps:
        cv2.imshow('A channel (LAB)', A)
        cv2.imshow('A Mask (Thresholded)', a_mask)
        cv2.imshow('Opening', opening)
        cv2.imshow('Distance Transform', dist_transform / dist_transform.max())
        cv2.imshow('Sure Foreground', sure_fg)
        cv2.imshow('Sure Background', sure_bg)
        cv2.imshow('Watershed Result', original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pig_count

if __name__ == "__main__":
    print(os.getcwd())
    frame = cv2.imread('data/test_frames/test_pigs_4.jpg')
    if frame is None:
        raise ValueError("Image not loaded correctly. Please check the file path.")
    pig_count = count_pigs_lab_watershed(frame, show_steps=True)
    print(f"Detected pigs: {pig_count}")
