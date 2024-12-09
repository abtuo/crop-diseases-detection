# Plant Disease Detection

This project implements a YOLOv8-based object detection system for identifying plant diseases.

## Project Structure

```
.
├── train.py           # Main training script
├── data.yaml          # YOLO configuration file
├── dataset/           # Processed dataset directory
│   ├── images/       # Images split into train/val/test
│   └── labels/       # YOLO format labels
├── Train.csv         # Training data annotations
├── Test.csv          # Test data information
└── images/           # Raw image files
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- pandas
- scikit-learn
- PIL
- PyYAML

## Installation

```bash
pip install ultralytics pandas scikit-learn pillow pyyaml
```

## Usage

1. Place your data files in the project root:
   - `Train.csv`: Training annotations
   - `Test.csv`: Test set information
   - `images/`: Directory containing all images

2. Run the training script:
```bash
python train.py
```

The script will:
- Preprocess the data and create YOLO format annotations
- Split the data into training and validation sets
- Train the YOLOv8 model
- Generate predictions on the test set
- Save the results in `submission.csv`

## Code Structure

### DataPreprocessor Class
Handles all data preparation tasks:
- Directory setup
- Data loading and splitting
- Image copying
- Label creation in YOLO format
- YAML configuration generation

### ModelTrainer Class
Manages model operations:
- Model training
- Validation
- Test set prediction

## Model Configuration

The default configuration uses:
- YOLOv8n (nano) model
- 100 epochs
- 1024x1024 image size
- Batch size of 8
- Early stopping patience of 5

You can modify these parameters in the `main()` function of `train.py`.
