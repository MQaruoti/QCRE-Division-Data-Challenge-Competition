
# Defect Detection Project

A simple defect detection pipeline that trains a model on training images and generates predictions for test images.

---

## Requirements

Install the required dependencies:

```bash
pip install opencv-python numpy pandas pillow
````

---

## Project Structure

```
DataFile/
│
├── train_data/          # Folder for training images
├── test_data/           # Folder for test images
├── trained_weights/     # Saved model weights
│
├── train.py             # Training script
├── detection.py         # Detection / inference script
└── ReadMe.md            # Project documentation
```

---

## Training

Run the training script:

```bash
python train.py
```

After training completes, the model weights will be saved to:

```
trained_weights/model_weights.npy
```

---

## Detection

1. Place the test images inside:

```
test_data/
```

2. Run the detection script:

```bash
python detection.py
```

The script will generate the prediction file:

```
test_detection.csv
```

---

## Output Format

The output CSV file contains the following columns:

```
image_id, defect, DT1, DT2, DT3
```

Where:

* **image_id** – Name or identifier of the image
* **defect** – Predicted defect label
* **DT1, DT2, DT3** – Detection metrics or prediction scores

```
```
