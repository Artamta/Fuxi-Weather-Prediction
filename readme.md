# 🌍 Fuxi: Transformers for Climate Forecasting

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20WIP-orange)](#)

**Under Development:**

> **Fuxi is a transformer-based pipeline for spatio-temporal climate forecasting.**
> It combines **Cube Embedding 🧊**, **Swin Transformer 🌐**, and **U-Transformer hierarchy ⬆⬇** to capture both local and global dependencies in climate data.

---

## 🚧 Project Status

- **Repository is in progress:**
  - Core Fuxi-inspired weather prediction model implemented and actively being trained.
  - Checkpoints, results, and new features will be updated as the project advances.

---

## 🌦️ Weather Prediction Model: Deep Learning for WeatherBench2

- **Dataset:** WeatherBench2, global reanalysis, 54x32 grid resolution.
- **Variables:**
  - 5 surface variables
  - 5 upper-air atmospheric variables
  - 13 pressure levels for upper-air variables
- **Input:** Previous 2 time steps (as multi-channel images/cubes)
- **Architecture Overview:**
  - **Cube Embedding:** Converts input sequence into compact feature cubes.
  - **U-Transformer Backbone:**
    - 48 repeated Swin Transformer blocks
    - Depths: [12, 12, 24]
    - Embedding dimension: 1536
  - **Fully Connected Layer:** Projects features to output channels.
  - **Bilinear Interpolation:** Upscales output to match target resolution.
- **Output:** Next time step prediction for all variables (multi-channel image).
- **Training:**
  - Latitude-weighted L1 loss for balanced global accuracy.
  - Model is currently training and under evaluation.

---

## 🔭 Future Work

- [ ] Increase spatial resolution of data and predictions
- [ ] Optimize and experiment with model architecture
- [ ] Full-fledged multi-step (autoregressive) prediction
- [ ] Fine-tuning and transfer learning for specific regions/events
- [ ] Advanced uncertainty estimation and ensemble methods

---

## 📂 Repository Structure

```
Fuxi-Weather-Prediction/
├── data/        # Input datasets (NetCDF, Xarray)
├── models/      # Cube embedding, transformer architectures
├── scripts/     # Training, evaluation, HPC jobs
├── configs/     # Config files for experiments
├── utils/       # Metrics, plotting, helpers
├── results/     # Logs, plots, checkpoints
└── README.md    # Project overview
```

---

## 🛠️ Methodology

**1. Data**

- Inputs: 5 surface variables, 5 upper-air variables (13 pressure levels), from WeatherBench2 (NetCDF/Xarray)
- Preprocessing: Temporal chunking, normalization, train/val/test splits

**2. Model Architecture**

- **Cube Embedding Layer:** Splits grid into spatio-temporal cubes
- **Swin Transformer Blocks:** Local windowed attention
- **U-Transformer Hierarchy:** Multi-scale global context
- **Prediction Head:** Multi-step forecasting

<p align="center">
  <img src="results/fuxi_architecture.png" alt="Fuxi Architecture" width="600"/>
</p>

**3. Training**

- Loss: Latitude-weighted L1 (for global balance)
- Optimizer: AdamW + Cosine Annealing LR
- Metrics: RMSE, R², ACC
- Hardware: HPC cluster (Slurm + SSH)

---

## 🖼️ Model Visualizations

### Swin Transformer

![Swin Transformer](plots/swin.png)

**Swin Transformer** divides the input grid into small windows and applies self-attention within each window. Windows are shifted between layers, allowing information to mix globally over several blocks.

- **Local Attention:** Each window focuses on local patterns.
- **Shifted Windows:** Overlapping windows in deeper layers help capture long-range dependencies.
- **Hierarchical:** The model downsamples and merges patches, building multi-scale representations.

---

### U-Net

![U-Net](plots/unet.jpg)

**U-Net** is a classic encoder-decoder architecture widely used for image segmentation and scientific data.

- **Encoder (Downsampling):** Compresses the input, extracting global features.
- **Decoder (Upsampling):** Restores the original resolution.
- **Skip Connections:** Connect encoder and decoder layers at the same scale, preserving fine details lost during downsampling.

---

### U-Transformer

![U-Transformer](plots/swin.png)

**U-Transformer** combines the U-Net’s multi-scale structure with transformer blocks (often Swin blocks).

- **Down Path:** Input is compressed through hierarchical transformer layers (like Swin).
- **Up Path:** Features are upsampled, with skip connections from earlier layers.
- **Multi-Scale Attention:** Captures both local and global dependencies, making it powerful for climate and scientific forecasting.

---

## 📊 Results (WIP)

| Fold | RMSE ↓ | R² ↑ | ACC ↑ |
| ---- | ------ | ---- | ----- |
| 1    | —      | —    | —     |
| 2    | —      | —    | —     |

**Training Loss Curve:**  
![Loss Curve](results/loss_curves.png)

**Sample Forecasts:**  
![Forecasts](results/sample_forecasts.png)

---

## 🚀 Quickstart

**1. Clone & Install**

```bash
git clone https://github.com/Artamta/Fuxi-Weather-Prediction.git
cd Fuxi-Weather-Prediction
conda create -n fuxi python=3.10
conda activate fuxi
pip install -r requirements.txt
```

**2. Train**

```bash
python scripts/train.py --config configs/default.yaml
```

**3. Evaluate**

```bash
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth
```

**4. Run on HPC**

```bash
sbatch scripts/slurm_train.sh
```

---

## 📌 Future Work

- Integrate full Fuxi pipeline
- Extend to multi-variable datasets (precipitation, NDVI, etc.)
- Compare with baselines (ConvLSTM, GNNs)
- Add transfer learning on ERA5/IMD data
- Release pre-trained weights

---

## 📚 References

- Wu et al., 2022 — Fuxi: A Transformer for Spatio-Temporal Forecasting in Climate Science
- Vaswani et al., 2017 — Attention is All You Need
- Liu et al., 2021 — Swin Transformer

---

💡 _This repository is under active development — contributions and feedback are welcome!_
