# 🌍 Fuxi: Transformers for Climate Forecasting

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Status](https://img.shields.io/badge/Status-Research%20WIP-orange)](#)

> **Fuxi is a transformer-based pipeline for spatio-temporal climate forecasting.**  
> It combines **Cube Embedding 🧊**, **Swin Transformer 🌐**, and **U-Transformer hierarchy ⬆⬇** to capture both local and global dependencies in climate data.

---

## ✨ Features

- 🧊 Cube Embedding for spatio-temporal patches
- 🌐 Hierarchical Transformers (Swin + U-Transformer)
- ⚡ HPC-ready (Slurm + SSH job submission)
- 📊 Metrics: RMSE, R², Anomaly Correlation Coefficient (ACC)
- 🔁 Multi-step climate forecasting
- 📦 Modular + Reproducible pipelines

---

## 🔬 Motivation

Climate systems are complex and multiscale. Traditional ML methods fail to capture **long-range temporal dependencies** and **regional correlations**.

**Fuxi** addresses this by:

- Embedding fields as **spatio-temporal cubes**
- Applying **hierarchical attention** to balance global vs local context
- Forecasting **extreme events** such as heatwaves & precipitation anomalies

---

## 📂 Repository Structure

Fuxi-Weather-Prediction/
│
├── data/ # Input climate datasets (NetCDF, Xarray)
├── models/ # Cube embedding, transformers, full Fuxi
├── scripts/ # Training, evaluation, HPC submission
├── configs/ # Config files (local & HPC runs)
├── utils/ # Metrics, plotting, helpers
├── results/ # Logs, plots, checkpoints
└── README.md # You are here 🚀

---

## 🛠️ Methodology

<details>
<summary><b>1. Data</b></summary>

- Input: Temperature, precipitation, NDVI in NetCDF/Xarray
- Preprocessing:
  - Temporal chunking
  - Normalization (z-score per variable)
  - Train/val/test by time slices
  </details>

<details>
<summary><b>2. Model Architecture</b></summary>

- **Cube Embedding Layer** 🧊 → splits grid into spatio-temporal cubes
- **Swin Transformer Blocks** 🌐 → local attention
- **U-Transformer Hierarchy** ⬆⬇ → global context
- **Prediction Head** → multi-step climate forecasting

<p align="center">
  <img src="results/fuxi_architecture.png" alt="Fuxi Architecture" width="600"/>
</p>
</details>

<details>
<summary><b>3. Training</b></summary>

- Loss: MSE / MAE
- Optimizer: AdamW + Cosine Annealing LR
- Metrics: RMSE, R², ACC
- Hardware: HPC cluster (Slurm + SSH)
</details>

---

## 📊 Results (Work in Progress)

| Fold | RMSE ↓ | R² ↑ | ACC ↑ |
| ---- | ------ | ---- | ----- |
| 1    | —      | —    | —     |
| 2    | —      | —    | —     |

📉 Training Loss Curve  
![Loss Curve](results/loss_curves.png)

🌡️ Sample Forecasts  
![Forecasts](results/sample_forecasts.png)

---

## 🚀 Quickstart

### 1️⃣ Clone

```bash
git clone https://github.com/Artamta/Fuxi-Weather-Prediction.git
cd Fuxi-Weather-Prediction

2️⃣ Install
conda create -n fuxi python=3.10
conda activate fuxi
pip install -r requirements.txt

3️⃣ Train
python scripts/train.py --config configs/default.yaml

4️⃣ Evaluate
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth

5️⃣ Run on HPC (Slurm)
sbatch scripts/slurm_train.sh

📌 Future Work

 Integrate full Fuxi pipeline

 Extend to multi-variable datasets (precip, NDVI, etc.)

 Compare with baselines (ConvLSTM, GNNs)

 Add transfer learning on ERA5/IMD data

 Release pre-trained weights

📚 References

Wu et al., 2022 — Fuxi: A Transformer for Spatio-Temporal Forecasting in Climate Science

Vaswani et al., 2017 — Attention is All You Need

Liu et al., 2021 — Swin Transformer

💡 This repository is under active development — contributions and feedback are welcome!




```
