# QCRE-Division-Data-Challenge-Competition
# Defect Detection Project

## Requirements

Install dependencies:

pip install opencv-python numpy pandas pillow

## Folder Structure

DataFile/
│
├── train_data
├── test_data
├── trained_weights
│
├── train.py
├── detection.py
└── ReadMe.md

## Training

Run:

python train.py

Weights will be saved to:

trained_weights/model_weights.npy

## Detection

Place test images inside:

test_data/

Run:

python detection.py

The script will generate:

test_detection.csv

## Output Format

image_id, defect, DT1, DT2, DT3
