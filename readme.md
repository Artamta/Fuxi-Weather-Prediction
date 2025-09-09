# 🌍 Fuxi: Transformers for Climate Forecasting

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20WIP-orange)](#)

> **Fuxi is a transformer-based pipeline for spatio-temporal climate forecasting.**
> It combines **Cube Embedding 🧊**, **Swin Transformer 🌐**, and **U-Transformer hierarchy ⬆⬇** to capture both local and global dependencies in climate data.

---

## ✨ Key Features

- **🧊 Cube Embedding:** Efficient spatio-temporal patching
- **🌐 Hierarchical Transformers:** Swin + U-Transformer for multi-scale attention
- **⚡ HPC-Ready:** Slurm & SSH job submission
- **📊 Metrics:** RMSE, R², Anomaly Correlation Coefficient (ACC)
- **🔁 Multi-step Forecasting:** Predict several future timesteps
- **📦 Modular & Reproducible:** Easy to extend and rerun

---

## 🔬 Motivation

Climate systems are complex and multiscale. Traditional ML methods struggle with **long-range dependencies** and **regional correlations**.

**Fuxi** solves this by:

- Embedding data as spatio-temporal cubes
- Using hierarchical attention for both global and local context
- Targeting extreme events (heatwaves, precipitation anomalies)

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

- Inputs: Temperature, precipitation, NDVI (NetCDF/Xarray)
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

- Loss: MSE / MAE
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

## 🧑‍💻 Transformer Implementation (From Scratch)

Your [`Tranformer/Transformer.py`](Tranformer/Transformer.py) file implements a full transformer model (encoder-decoder) from scratch using PyTorch.

**Key Components:**

- **SelfAttention:**  
  Splits input into multiple heads, computes attention scores, and mixes information from all tokens.
- **TransformerBlock:**  
  Applies self-attention, adds a feed-forward neural network, and uses LayerNorm and residual connections for stability.
- **Encoder:**  
  Embeds input tokens and their positions, stacks several TransformerBlocks, and outputs contextual representations.
- **Decoder:**  
  Embeds target tokens and positions, stacks DecoderBlocks (masked self-attention + cross-attention to encoder output), and predicts the next token at each position.
- **Masks:**  
  Source mask hides padding tokens; target mask enforces causal (left-to-right) prediction.
- **Full Transformer:**  
  Combines encoder and decoder for tasks like translation, sequence prediction, or forecasting.

**How It Works:**

1. **Input Preparation:**  
   Source and target sequences are embedded and given positional information.
2. **Encoding:**  
   The encoder processes the source sequence, building rich representations.
3. **Decoding:**  
   The decoder generates the output sequence, one token at a time, using both its own previous outputs and the encoder’s memory.
4. **Attention Mechanism:**  
   Allows the model to focus on relevant parts of the input when generating each output token.
5. **Training:**  
   Uses teacher forcing: the decoder receives the true previous tokens during training.

**Why This Matters:**

- **From Scratch:** No external transformer libraries; all logic is implemented manually.
- **Flexible:** Can be adapted for text, images, or climate grids.
- **Educational:** Shows a clear understanding of transformer internals.

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
