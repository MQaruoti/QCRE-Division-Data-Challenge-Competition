import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from glob import glob

def log(msg):
    print(f"[LOG] {msg}")

# Detect holes
def detect_holes(image_path, dp=1.2, min_dist=20, param1=50, param2=15, min_radius=3, max_radius=15):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    detected = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for x, y, r in circles:
            detected.append({"x": int(x), "y": int(y), "r": int(r)})

    return detected


# Cluster holes into regions
def cluster_regions(holes, expected_regions=4):

    if len(holes) == 0:
        return []

    holes_sorted = sorted(holes, key=lambda h: h["x"])
    step = max(1, len(holes_sorted) // expected_regions)

    regions = []

    for i in range(expected_regions):

        group = holes_sorted[i*step:(i+1)*step]

        if not group:
            continue

        x_min = min(h["x"] - h["r"] for h in group)
        x_max = max(h["x"] + h["r"] for h in group)
        y_min = min(h["y"] - h["r"] for h in group)
        y_max = max(h["y"] + h["r"] for h in group)

        regions.append({
            "box": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max},
            "holes": group
        })

    return regions


# Analyze defects
def analyze_defects(region, grid_rows=5, grid_cols=5):

    holes = region["holes"]
    box = region["box"]

    DT1 = 0
    DT2 = 0
    DT3 = 0

    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]

    expected_dx = width / (grid_cols - 1)
    expected_dy = height / (grid_rows - 1)

    # Missing holes
    if len(holes) < grid_rows * grid_cols:
        DT1 = 1

    # Touching holes
    for i in range(len(holes)):
        for j in range(i+1, len(holes)):

            dist = np.hypot(
                holes[i]["x"] - holes[j]["x"],
                holes[i]["y"] - holes[j]["y"]
            )

            if dist < min(expected_dx, expected_dy) * 0.7:
                DT2 = 1
                break

        if DT2:
            break

    # Out of bounds
    for h in holes:
        if h["x"] < box["xmin"] or h["x"] > box["xmax"] or h["y"] < box["ymin"] or h["y"] > box["ymax"]:
            DT3 = 1
            break

    return DT1, DT2, DT3


# Process single image
def process_image(image_path):

    image_id = os.path.basename(image_path)

    holes = detect_holes(image_path)

    regions = cluster_regions(holes, expected_regions=4)

    DT1_total = 0
    DT2_total = 0
    DT3_total = 0

    for r in regions:

        DT1, DT2, DT3 = analyze_defects(r)

        DT1_total = max(DT1_total, DT1)
        DT2_total = max(DT2_total, DT2)
        DT3_total = max(DT3_total, DT3)

    defect = 1 if (DT1_total or DT2_total or DT3_total) else 0

    return {
        "image_id": image_id,
        "defect": defect,
        "DT1": DT1_total,
        "DT2": DT2_total,
        "DT3": DT3_total
    }


# Process folder
def process_folder(folder_path):

    image_paths = glob(os.path.join(folder_path, "*.jpg")) + \
                  glob(os.path.join(folder_path, "*.png"))

    results = []

    for img_path in image_paths:

        row = process_image(img_path)

        results.append(row)

    df = pd.DataFrame(results)

    return df


if __name__ == "__main__":

    folder_path = "test_data"   # required competition folder

    results_df = process_folder(folder_path)

    print(results_df)

    # Save EXACT required file
    results_df.to_csv("test_detection.csv", index=False)
