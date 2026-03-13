import cv2
import numpy as np
import pandas as pd
import os
from glob import glob

def detect_holes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=15
    )

    holes = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for x, y, r in circles:
            holes.append((x, y, r))

    return holes


def analyze_defects(holes):

    DT1 = 0
    DT2 = 0
    DT3 = 0

    expected_holes = 25

    if len(holes) < expected_holes:
        DT1 = 1

    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            dist = np.sqrt((holes[i][0] - holes[j][0])**2 +
                           (holes[i][1] - holes[j][1])**2)
            if dist < 10:
                DT2 = 1
                break
        if DT2:
            break

    for h in holes:
        if h[0] < 0 or h[1] < 0:
            DT3 = 1
            break

    return DT1, DT2, DT3


def process_images():

    test_folder = "test_data"
    images = glob(os.path.join(test_folder, "*.png")) + \
             glob(os.path.join(test_folder, "*.jpg"))

    results = []

    for img_path in images:

        image_id = os.path.basename(img_path)

        holes = detect_holes(img_path)

        DT1, DT2, DT3 = analyze_defects(holes)

        defect = 1 if (DT1 or DT2 or DT3) else 0

        results.append({
            "image_id": image_id,
            "defect": defect,
            "DT1": DT1,
            "DT2": DT2,
            "DT3": DT3
        })

    df = pd.DataFrame(results)

    df.to_csv("test_detection.csv", index=False)


if __name__ == "__main__":
    process_images()
