# SAGENet-Acoustic-Echo-Based-3D-Depth-Estimation-with-Sparse-Angular-Query-and-Refined-Geometric-Cues


Official implementation of SAGENet: Acoustic Echo-Based 3D Depth Estimation with Sparse Angular Query and Refined Geometric Cues.
The code is coming soon!

## Project Overview

This project aims to estimate scene depth information from binaural audio signals. The proposed method integrates refined 2D geometric cues and employs angle spectrum peaks to guide feature attention, enhancing depth estimation performance. 

## File Structure

    EchoDepthformer/
    ├── config/
    │   └── config.yaml          # Configuration file
    ├── data_loader/
    │   └── custom_dataset_data_loader.py  # Data loader
    ├── models/
    │   ├── audioVisual_model.py  # Audio-visual model
    │   ├── models.py             # Model builder
    │   └── PointNet.py           # PointNet model
    ├── utils/
    │   └── Opt_.py               # Configuration option parser
    ├── train.py                  # Training script
    ├── inference_visualize.py    # Inference and visualization script
    ├── EchoPCL.py                    # Echo PCL algorithm implementation
    ├── README.md                 # Project documentation
    └── requirements.txt          # Dependencies file

## Installation

Follow the steps below to install the required dependencies:

1. Clone the repository:

2. Create and activate a virtual environment (optional):

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows users, use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

Run the [train.py](http://_vscodecontentref_/3) script to train the model:

    ```bash
    python train.py
    ```
