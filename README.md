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

### **Usage**
--------------

#### **Step 1: Run `segmentation_cellpose.py` to See the Initial Segmentation**
Before comparing the segmented cells with manually annotated shapes, you need to first segment the bacterial cells using **Cellpose**.

- **Description**: This script will perform segmentation of bacterial cells in the image using the **Cellpose Cyto model**. The output will be a **segmented mask** for the bacterial cells.

- **Command**:
    ```bash
    python segmentation_cellpose.py
    ```

- **Input**: The script expects an image in **CZI** format (e.g., `WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi`).
- **Output**: The output is a **segmented mask** of the bacterial cells and optionally a **3D visualization** of the segmentation.

Make sure your **image files** (e.g., `.czi` format) are available in the same directory, or provide the correct path in the script.

---

#### **Step 2: Run `Criteria.py` to Compare the Segmented Cell Shapes**
Once you’ve segmented the cells, the next step is to compare the segmented cell shapes with the **manually annotated ground truth**.

- **Description**: This script will evaluate the performance of the segmentation by comparing the segmented cell shapes with the **manually annotated shapes**. It calculates performance metrics such as **Dice score**, **IoU (Intersection over Union)**, and **accuracy**.

- **Command**:
    ```bash
    python Criteria.py
    ```

- **Input**: The script uses the **segmented mask** from `segmentation_cellpose.py` and the **ground truth annotations** (stored in a `.zip` file).
- **Output**: The output includes the **evaluation metrics**: Dice score, IoU, and accuracy.

---

#### **Step 3: Run `3d_segment.py` to Visualize the 2D Segmentation in 3D**
After segmenting and evaluating the cells, the final step is to visualize how the **2D segmented shapes** are reconstructed in **3D**.

- **Description**: This script uses the **Cellpose segmentation results** to fit **ellipsoids** to each segmented cell. It then visualizes these ellipsoids in 3D, allowing you to see how the 2D segmentation translates to a 3D structure.

- **Command**:
    ```bash
    python 3d_segment.py
    ```

- **Input**: The script uses the **segmented mask** from `segmentation_cellpose.py`.
- **Output**: The output is a **3D visualization** of the bacterial cells using **ellipsoids**, showing how the 2D segmentation is represented in 3D.

---

### **Make Sure Your Files Are in the Correct Location**
For all the scripts, ensure that your **image files** (e.g., `.czi` format) and **ground truth annotations** (e.g., `.zip` with ROI data) are available in the same directory or specify the correct paths in the script.

---



