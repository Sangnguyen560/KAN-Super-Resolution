# Super-Resolution Experiments on MNIST and FashionMNIST

This project implements and compares various super-resolution models on MNIST and FashionMNIST datasets. The models upscale low-resolution (8x8) images to high-resolution (64x64) images, with performance evaluated using PSNR and SSIM metrics. The models implemented include CNN, FastKAN, KAN-SR, and MLP.

## Features
- **Multiple Models**: Implements four different architectures for super-resolution:
  - CNN (Convolutional Neural Network)
  - FastKAN (Fast Kolmogorov-Arnold Network)
  - KAN-SR (Kolmogorov-Arnold Network with Spline)
  - MLP (Multi-Layer Perceptron)
- **Datasets**: Experiments on both MNIST and FashionMNIST datasets.
- **Super-Resolution**: Converts 8x8 images to 64x64 with quality evaluation.
- **Comprehensive Outputs**: Saves training/validation metrics, model parameters, and visual comparisons.
- **Dataset Splits**: Includes separate train, validation, and test sets.

## Directory Structure
- **data/**: Stores MNIST and FashionMNIST datasets (automatically downloaded).
- **train_data_set/**: Saved training datasets for each model.
- **val_data_set/**: Saved validation datasets for each model.
- **test_data_set/**: Saved test datasets for each model.
- **Output_FashionMNIST/**: Output directory for FashionMNIST experiments.
  - `output_cnn/`, `output_fastkan/`, `output_kan_sr/`, `output_mlp/`: Subdirectories for each model containing:
    - Metrics, plots, and visual results.
- **Output_MNIST/**: Output directory for MNIST experiments.
  - Same structure as `Output_FashionMNIST/`.
- **Test_FashionMNIST/**: Test scripts for FashionMNIST.
  - `CNN_FashionMNIST.py`, `FastKAN_FashionMNIST.py`, `KAN_SR_FashionMNIST.py`, `MLP_FashionMNIST.py`: Scripts for each model.
- **Test_MNIST/**: Test scripts for MNIST.
  - `CNN_MNIST.py`, `FastKAN_MNIST.py`, `KAN_SR_MNIST.py`, `MLP_MNIST.py`: Scripts for each model.
- **best_model_kan_sr.pt**: Saved best model weights for KAN-SR.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation
Set up a virtual environment (optional but recommended) and install the required libraries.

### 1. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Required Libraries
Install each library individually to ensure compatibility.

- **PyTorch**: For deep learning framework.
  ```bash
  pip install torch
  ```
  If you have a GPU, install a CUDA-compatible version (check PyTorch's website for your CUDA version):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118  # Example for CUDA 11.8
  ```

- **torchvision**: For dataset loading (MNIST, FashionMNIST).
  ```bash
  pip install torchvision
  ```

- **NumPy**: For numerical computations.
  ```bash
  pip install numpy
  ```

- **Matplotlib**: For plotting and visualization.
  ```bash
  pip install matplotlib
  ```

- **scikit-image**: For PSNR and SSIM metrics.
  ```bash
  pip install scikit-image
  ```

- **tqdm**: For progress bars during training.
  ```bash
  pip install tqdm
  ```

- **prettytable**: For displaying model parameter tables.
  ```bash
  pip install prettytable
  ```

## Usage
Follow these steps to run the super-resolution experiments for each model on MNIST and FashionMNIST datasets.

### 1. Clone the Repository or Set Up the Directory
Ensure your directory matches the structure described above. If you don't have the scripts, you can organize them accordingly or clone the repository if available.

### 2. Run Experiments for MNIST
Navigate to the project directory and run the scripts for each model on the MNIST dataset.

- **CNN Model**:
  ```bash
  python Test_MNIST/CNN_MNIST.py
  ```
  Outputs will be saved in `Output_MNIST/output_cnn/`.

- **FastKAN Model**:
  ```bash
  python Test_MNIST/FastKAN_MNIST.py
  ```
  Outputs will be saved in `Output_MNIST/output_fastkan/`.

- **KAN-SR Model**:
  ```bash
  python Test_MNIST/KAN_SR_MNIST.py
  ```
  Outputs will be saved in `Output_MNIST/output_kan_sr/`.

- **MLP Model**:
  ```bash
  python Test_MNIST/MLP_MNIST.py
  ```
  Outputs will be saved in `Output_MNIST/output_mlp/`.

### 3. Run Experiments for FashionMNIST
Run the scripts for each model on the FashionMNIST dataset.

- **CNN Model**:
  ```bash
  python Test_FashionMNIST/CNN_FashionMNIST.py
  ```
  Outputs will be saved in `Output_FashionMNIST/output_cnn/`.

- **FastKAN Model**:
  ```bash
  python Test_FashionMNIST/FastKAN_FashionMNIST.py
  ```
  Outputs will be saved in `Output_FashionMNIST/output_fastkan/`.

- **KAN-SR Model**:
  ```bash
  python Test_FashionMNIST/KAN_SR_FashionMNIST.py
  ```
  Outputs will be saved in `Output_FashionMNIST/output_kan_sr/`.

- **MLP Model**:
  ```bash
  python Test_FashionMNIST/MLP_FashionMNIST.py
  ```
  Outputs will be saved in `Output_FashionMNIST/output_mlp/`.

### 4. Check Outputs
After running each script, check the respective output directories (`Output_MNIST/` or `Output_FashionMNIST/`) for:
- Training progress logs with loss, PSNR, and SSIM metrics.
- Model parameter counts and unused parameter detection.
- Plots (e.g., PSNR over epochs), metrics files (e.g., `test_metrics.txt`), and visual comparisons (e.g., `results_kan_sr.pdf`).

## Code Overview
- **Models**: Each script (`CNN_*.py`, `KAN_SR_*.py`, etc.) implements a specific model for super-resolution.
- **Dataset**: Custom dataset class creates low-res (8x8) and high-res (64x64) pairs from MNIST/FashionMNIST.
- **Training**: Uses Adam optimizer, MSE loss, and early stopping based on validation PSNR.
- **Evaluation**: Computes PSNR and SSIM for validation and test sets.
- **Visualization**: Saves PDFs with low-res, super-resolved, and high-res images for comparison.

## Notes
- Models are trained on a subset of digits/classes (e.g., digits 9, 7, 6, 1 for MNIST) to reduce dataset size.
- Training stops early if validation PSNR does not improve for 5 epochs.
- Outputs are organized by dataset and model in `Output_MNIST/` and `Output_FashionMNIST/`.
- Ensure sufficient disk space for datasets, model weights, and output files.

## License
This project is licensed under the MIT License.