# bimap_p2
# BIMAP-P2: Characterizing Bacteria Using Shape Descriptors

This project focuses on characterizing bacterial cells through image segmentation, growth rate analysis, and 3D shape reconstruction. The goal is to provide an automated and quantifiable solution for understanding bacterial behavior, including peptide distribution, growth rates, and cellular morphology. This project employs advanced image processing techniques such as **Cellpose segmentation**, **3D reconstruction**, and evaluation metrics for performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Scripts](#scripts)
  - [Criteria.py](#criteriapy)
  - [3d_segment.py](#3d_segmentpy)
  - [segmentation_cellpose.py](#segmentation_cellposepy)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview
Bacterial shape analysis is crucial for understanding bacterial behavior, including how they interact with the host and their growth patterns. In this project, we focus on:
- **Segmentation** of bacterial cells using the **Cellpose Cyto model**.
- **Evaluation of segmentation accuracy** using metrics like **Dice Score** and **IoU**.
- **3D reconstruction** of bacterial cells using **Vedo** for visualizing their shapes.

## Technologies Used
- **Python**: For scripting and automation.
- **Cellpose**: For cell segmentation.
- **Vedo**: For 3D visualization and ellipsoid fitting.
- **Scikit-learn**: For evaluation metrics (IoU, Dice score).
- **Scipy**: For image processing and morphological operations.
- **Matplotlib**: For visualization of results.

## Scripts

### **Criteria.py**
- **Description**: This script loads microscopy images (in `.czi` format), segments bacterial cells using the **Cellpose Cyto model**, and evaluates the segmentation results using metrics such as **Dice score**, **IoU**, **F1 score**, and **accuracy**.
  
### **3d_segment.py**
- **Description**: This script performs 3D reconstruction of bacterial cells from 2D segmented images. It uses the **Cellpose Cyto model** for segmentation and fits **ellipsoids** to each bacterial cell to visualize their 3D shapes.

### **segmentation_cellpose.py**
- **Description**: This script handles the segmentation of bacterial cells using **Cellpose** and post-processes the segmented masks. It also includes an optional step for **3D reconstruction** using fitted ellipsoids.

## Installation

To get started with this project, you need to have **Python** installed on your system. The following libraries are required to run the scripts:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/<username>/<repository-name>.git
    cd <repository-name>
    ```

2. **Install the required dependencies**:

    It's recommended to use a `virtual environment` to manage dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Mac/Linux
    venv\Scripts\activate  # For Windows
    ```

    Then, install the necessary dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can manually install the libraries with:

    ```bash
    pip install cellpose vedo scipy numpy matplotlib scikit-learn czifile
    ```

## Usage

### Running `Criteria.py`:
```bash
python Criteria.py

### Running `3d_segment.py`
This script performs 3D reconstruction of bacterial cells from segmented images. It uses the **Cellpose Cyto model** to segment cells and fits **ellipsoids** to visualize their 3D shapes.

#### Command:
```bash
python 3d_segment.py
