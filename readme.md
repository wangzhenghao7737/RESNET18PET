# ResNet18-Based Cat vs Dog Classifier (Optional SEBlock)

This project implements a binary image classification model (e.g., cat vs. dog) based on **ResNet18**. It optionally integrates a **Squeeze-and-Excitation Block (SEBlock)** to enhance feature channel attention.

------

## Project Structure

```
.
├── ConfusionMatrix/              # Confusion matrix images from evaluation
├── dataset/                      # Dataset containing training and testing images
├── model/                        # Trained model (.pth) files
├── MyModule/                     # Contains SEBlock module
├── PerformanceResult/           # Accuracy and loss plots during training
├── autodll.md                   # Instructions for using the AutoDL platform
├── best_gln_model.pth           # Best model weights (optional)
├── cc.jpg                       # External image for prediction
├── data_partitioning.py         # Script to split dataset
├── model.py                     # ResNet18 and ResidualBlock definitions
├── model_test_one_picture.py    # Predict a single image
├── model_train.py               # Training script
├── requirements.txt             # Python dependencies
├── result_of_test_py.txt        # Output log of test.py
├── test.py                      # Model evaluation & confusion matrix generation
```

------

##  Installation

Make sure Python 3.9+ is installed. Try to use the Windows system and the Anaconda environment.

```
# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch 
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

------

## Quick Start

### 1. Dataset Preparation

Organize your dataset like this (for example, cat vs. dog):

```
dataset/
└── train/
    ├── cat/
    │   ├── xxx.jpg
    │   └── ...
    ├── dog/
        ├── yyy.jpg
        └── ...
```

You can optionally use `data_partitioning.py` to automate the dataset split.

------

### 2. Train the Model

```bash
python model_train.py --epochs 50 --batch_size 64 --lr 0.0005 --use_se
```

**Arguments**:

| Argument       | Description                  | Default |
| -------------- | ---------------------------- | ------- |
| `--epochs`     | Number of training epochs    | `50`    |
| `--batch_size` | Batch size for training      | `32`    |
| `--lr`         | Learning rate                | `0.001` |
| `--use_se`     | Use SEBlock attention module | `False` |

After training, the best model is saved in the `model/` directory, and loss/accuracy plots are saved to `PerformanceResult/`.

------

### 3. Evaluate the Model

```bash
python test.py
```

This will:

- Load the trained model
- Calculate test accuracy
- Generate and save a **confusion matrix** to `ConfusionMatrix/`

------

### 4. Predict on a Single Image

```
python model_test_one_picture.py --img other/another_image.jpg --model model/custom_model.pth
```

------

##  Visualization

**Training Accuracy & Loss** (in `PerformanceResult/`):

**Confusion Matrix** (in `ConfusionMatrix/`):

------

##  Module Descriptions

| File/Folder                 | Description                                   |
| --------------------------- | --------------------------------------------- |
| `model.py`                  | ResNet18 and residual block architecture      |
| `MyModule/SEBlock.py`       | Implementation of SEBlock attention           |
| `model_train.py`            | Model training logic                          |
| `test.py`                   | Model evaluation, confusion matrix generation |
| `data_partitioning.py`      | Utility for dataset splitting                 |
| `model_test_one_picture.py` | Test a single image                           |

------

##  Model Architecture

- Backbone: ResNet18 
- Optional: Channel-wise SEBlock for attention

------

## AutoDL Platform

Refer to `autodll.md` for guidance on how to deploy and run this project on **AutoDL** platform.

------

