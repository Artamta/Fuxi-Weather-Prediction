# Vision Transformer (ViT) and Full Transformer Implementation on MNIST

This repository contains a **from-scratch implementation of the Transformer architecture** (encoder-decoder, as in the original "Attention is All You Need" paper) in PyTorch, as well as a Vision Transformer (ViT) for image classification on MNIST. All code is modular, readable, and ready for experimentation or extension.

---

## 📚 What is a Transformer?

The Transformer is a neural network architecture introduced by Vaswani et al. (2017) that relies entirely on **attention mechanisms** to draw global dependencies between input and output. Unlike RNNs or CNNs, it processes sequences in parallel and can model long-range relationships efficiently.

### **Key Components:**

- **Multi-Head Self Attention:** Allows the model to focus on different parts of the input sequence simultaneously.
- **Positional Encoding:** Since the model has no recurrence or convolution, positional encodings inject information about the order of the sequence.
- **Encoder:** Processes the input sequence and produces contextual representations.
- **Decoder:** Generates the output sequence, attending to both previous outputs and the encoder's memory.
- **Feedforward Layers & LayerNorm:** Stabilize and enrich the representations.
- **Masking:** Ensures the decoder only attends to previous tokens during training (teacher forcing).

---

## 🛠️ Implementation Details

### **1. Multi-Head Self Attention**

- Splits the input into multiple "heads" for parallel attention.
- Computes attention scores (scaled dot-product) between all tokens.
- Mixes information from all positions in the sequence.

### **2. Transformer Block**

- Each block consists of a multi-head attention layer, followed by a feedforward network.
- Residual connections and layer normalization are used for stability.

### **3. Encoder**

- Embeds tokens and their positions.
- Stacks several transformer blocks to build deep contextual representations.

### **4. Decoder**

- Similar to the encoder, but includes masked self-attention and cross-attention to the encoder output.
- Generates the output sequence one token at a time.

### **5. Full Transformer Model**

- Combines encoder and decoder for tasks like translation, sequence prediction, or forecasting.
- Implements source and target masks for padding and causality.

---

## 🖼️ Architecture Diagrams

_Add your images here for clarity:_

- `images/transformer_architecture.png` — Full transformer encoder-decoder
- `images/self_attention.png` — Multi-head self-attention mechanism
- `images/vit_patch_embedding.png` — Vision Transformer patch embedding

---

## 🚀 How to Use

### **Vision Transformer on MNIST**

- The ViT model splits each MNIST image into small patches, embeds them, and processes the sequence with transformer blocks.
- After training, the model achieves strong accuracy on MNIST, demonstrating the effectiveness of transformer architectures for image tasks.

### **Full Transformer (Text/Sequence Tasks)**

- The `Transformer.py` implements the full encoder-decoder transformer as in the original paper.
- You can adapt this for text translation, time series forecasting, or other sequence-to-sequence problems.

---

## 📦 Repository Structure

```
transformer/
├── vit.ipynb           # Vision Transformer notebook for MNIST
├── Transformer.py      # Full transformer implementation (encoder-decoder)
├── images/             # Architecture diagrams and attention visualizations
└── README.md           # This file
```

---

## 🧑‍💻 Example Usage

**Vision Transformer (ViT) on MNIST:**

- See `vit.ipynb` for a step-by-step notebook.
- Uses patch embedding, transformer blocks, and a classifier head.

**Full Transformer:**

- See `Transformer.py` for the encoder-decoder implementation.
- Includes a test section showing how to instantiate and run the model.

---

## ⚡ Device Support

- Runs efficiently on Mac (MPS/Metal), CUDA, or CPU.
- All tensors and models are moved to the appropriate device automatically.

---

## 📊 Results

- Training loss curves and validation accuracy are plotted in the notebook.
- The transformer models achieve high accuracy on MNIST and are easy to extend for deeper models or other datasets.

---

## 📚 References

- Vaswani et al., 2017 — [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Dosovitskiy et al., 2020 — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

---

## 💡 Notes

- All code is written from scratch, with no external transformer libraries.
- Modular design makes it easy to experiment with different architectures, depths, and tasks.
- Add your own images to the `images/` folder and reference them in this README for clarity.

---

## 🙌 Contributing

Contributions, suggestions, and feedback are welcome!
