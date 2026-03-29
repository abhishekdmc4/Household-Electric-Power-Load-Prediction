# Household Electric Power Load Prediction: Resource-Efficient Transformers for the Edge

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Solving the "Intelligence-Compute Gap" in Smart Grids: This repository contains the source code and experimental pipeline for deploying state-of-the-art deep learning architectures directly onto resource-constrained smart meter hardware.

By engineering a lightweight attention architecture (Enhanced TinyTFT) and applying aggressive TinyML compression pipelines, this project shifts the forecasting paradigm from "Cloud-First" to "Edge-Only," providing a mathematically validated blueprint for autonomous, privacy-preserving residential load balancing.

---

## Table of Contents
* Key Features
* Dataset
* Repository Structure
* Installation & Setup
* Usage & Execution Flow
* Core Results
* Technologies Used
* Acknowledgments

---

## Key Features

* Direct Multi-Step Forecasting: Bypasses the "recursive wall" (error compounding) seen in classical models by projecting 24h, 72h, and 168h horizons in a single forward pass.
* The Enhanced TinyTFT: A custom-built transformer variant with mathematically downscaled embedding dimensions (d_model = 32), 4 attention heads, and 2 encoder layers designed specifically to prevent SRAM out-of-memory errors on microcontrollers.
* TinyML Compression Pipeline: Implements 30% L1 Unstructured Magnitude Pruning and Dynamic INT8 Quantization to shrink the physical memory footprint of the models by ~75% with statistically negligible predictive decay.
* CPU-Bound Edge Simulation: All inference timing and hardware profiling are strictly simulated in a single-threaded CPU environment using torch.no_grad() to accurately reflect the execution latency of ARM Cortex-M/RISC-V embedded systems.
* Pareto-Optimal Edge Benchmarking: Evaluates models across two violently opposed metrics: Predictive Accuracy (MASE/RMSE) vs. Hardware Latency (Logarithmic Seconds).

---

## Dataset

The experiments utilize the open-source UCI Individual Household Electric Power Consumption Dataset.

* Scale: Contains over 2.07 million minute-resolution observations of a single residence in France.
* Preprocessing: Missing values are imputed via linear interpolation, data is down-sampled to 1-hour resolutions to filter stochastic appliance noise, and cyclical temporal features (sine/cosine transformations) are engineered to capture diurnal and weekly seasonality.
* Sliding Window: A strict 168-hour (1 week) historical look-back window is used.

---

## Repository Structure

    data/                   # Directory for the UCI dataset (not tracked in git)
    images/                 # Charts, Pareto frontiers, and architecture diagrams
    Source code.ipynb       # Main Jupyter Notebook containing the end-to-end pipeline
    requirements.txt        # Python dependencies
    LICENSE                 # MIT License
    README.md               # Project documentation

---

## Installation & Setup

1. Clone the repository:
    git clone https://github.com/yourusername/TinyTFT-Edge-Forecasting.git
    cd TinyTFT-Edge-Forecasting

2. Create a virtual environment (Recommended):
    python -m venv venv
    source venv/bin/activate

3. Install dependencies:
    pip install -r requirements.txt

4. Download the Dataset:
    Download the UCI dataset, extract it, and place household_power_consumption.txt into the data/ directory.

---

## Usage & Execution Flow

Launch the Jupyter environment to execute the pipeline:
    jupyter notebook "Source code.ipynb"

Pipeline Steps Included in the Notebook:
1. Data Preprocessing: Resampling, Z-score scaling, and 3D tensor sequence generation.
2. Unconstrained Training: Deep learning models are trained using the AdamW optimizer, MSE loss, and ReduceLROnPlateau scheduling.
3. Post-Training Compression: Extraction of trained weights, application of torch.nn.utils.prune, and dynamic quantization via torch.ao.quantization.
4. Hardware Profiling: Edge simulation evaluating MASE, RMSE, execution latency, and parameter footprint.

---

## Core Results

The Optimized TinyTFT (30% Pruned + INT8) emerged as the definitive Pareto-optimal winner for edge deployment at the 168-hour horizon.

| Model | Architecture | 168h MASE | Latency (ms) | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| ARIMA | Classical Recursive | 0.8151 | 58,070 ms | < 1 KB |
| XGBoost | Decision Tree | 0.8872 | 185 ms | ~ 1 MB |
| LSTM (Uncompressed) | RNN | 0.6830 | 4.90 ms | ~ 2.5 MB |
| TinyTFT (Proposed) | Compressed Attention | 0.6773 | 3.50 ms | 1.56 MB |

Note: A Diebold-Mariano (DM) test confirmed the TinyTFT's predictive superiority over all baselines with strict statistical significance (p < 0.05).

---

## Technologies Used
* Core Language: Python 3.12
* Deep Learning Framework: PyTorch
* Machine Learning & Stats: Scikit-Learn, Statsmodels, XGBoost
* Data Processing: Pandas, NumPy
* Visualization: Matplotlib, Seaborn

---

## Acknowledgments
* Author: Abhishek Jain
* Institution: Liverpool John Moores University (MSc. Data Science)
* Special thanks to Dr. Prakash Kene and the faculty at LJMU for their guidance throughout this master's dissertation research.
