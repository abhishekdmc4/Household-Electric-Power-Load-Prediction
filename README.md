# ⚡ Household Electric Power Load Prediction: Resource-Efficient Transformers for the Edge

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Solving the "Intelligence-Compute Gap" in Smart Grids:** This repository contains the source code and experimental pipeline for deploying state-of-the-art deep learning architectures directly onto resource-constrained smart meter hardware.

By engineering a lightweight attention architecture (**Enhanced TinyTFT**) and applying aggressive TinyML compression pipelines, this project shifts the forecasting paradigm from "Cloud-First" to "Edge-Only," providing a mathematically validated blueprint for autonomous, privacy-preserving residential load balancing.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution Flow](#-usage--execution-flow)
- [Core Results](#-core-results)
- [Technologies Used](#-technologies-used)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Key Features

* **Direct Multi-Step Forecasting:** Bypasses the "recursive wall" (error compounding) seen in classical models by projecting 24h, 72h, and 168h horizons in a single forward pass.
* **The Enhanced TinyTFT:** A custom-built transformer variant with mathematically downscaled embedding dimensions (`d_model = 32`), 4 attention heads, and 2 encoder layers designed specifically to prevent SRAM out-of-memory errors on microcontrollers.
* **TinyML Compression Pipeline:** Implements **30% L1 Unstructured Magnitude Pruning** and **Dynamic INT8 Quantization** to shrink the physical memory footprint of the models by ~75% with statistically negligible predictive decay.
* **CPU-Bound Edge Simulation:** All inference timing and hardware profiling are strictly simulated in a single-threaded CPU environment using `torch.no_grad()` to accurately reflect the execution latency of ARM Cortex-M/RISC-V embedded systems.
* **Pareto-Optimal Edge Benchmarking:** Evaluates models across two violently opposed metrics: Predictive Accuracy (MASE/RMSE) vs. Hardware Latency (Logarithmic Seconds).

---

## 📊 Dataset

The experiments utilize the open-source **[UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)**.

* **Scale:** Contains over 2.07 million minute-resolution observations of a single residence in France.
* **Preprocessing:** Missing values are imputed via linear interpolation, data is down-sampled to 1-hour resolutions to filter stochastic appliance noise, and cyclical temporal features (sine/cosine transformations) are engineered to capture diurnal and weekly seasonality.
* **Sliding Window:** A strict 168-hour (1 week) historical look-back window is used.

---

## 📁 Repository Structure

```text
├── data/                   # Directory for the UCI dataset (not tracked in git)
├── images/                 # Charts, Pareto frontiers, and architecture diagrams
├── Source code.ipynb       # Main Jupyter Notebook containing the end-to-end pipeline
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation
