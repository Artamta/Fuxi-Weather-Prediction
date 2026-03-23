# FuXi Pre-training Setup Guide

Complete training setup for FuXi weather forecasting model with paper-faithful implementation.

## 📋 Overview

This setup provides:
- **Direct Zarr reading** (zero data copying, instant startup)
- **Multi-GPU training** via PyTorch DistributedDataParallel
- **Paper-faithful** pre-training (single-step prediction, 40k iterations)
- **Automatic checkpointing** and resume capability
- **SLURM job submission** for hyperparameter sweeps

## 🗂️ File Structure

```
your_project/
├── fuxi_train.py          # Main training script
├── fuxi_sweep.sh          # SLURM hyperparameter sweep
├── model.py               # Your FuXi model definition
├── blocks.py              # (optional) Model building blocks
├── Models/                # Output directory (created automatically)
│   ├── pretrain_*/        # Individual experiment folders
│   │   ├── best.pt        # Best checkpoint
│   │   ├── last.pt        # Latest checkpoint (for resume)
│   │   ├── config.json    # Training configuration
│   │   ├── metrics.json   # Final metrics
│   │   └── Plots/         # Visualizations
└── logs/                  # SLURM job logs
```

## 🔧 Setup Instructions

### 1. Verify Your Model Files

The training script imports your FuXi model with:
```python
from model import FuXi
```

**Required:** Your model class must match this signature:
```python
class FuXi(nn.Module):
    def __init__(
        self,
        num_variables: int,      # Number of input/output channels
        embed_dim: int,          # Embedding dimension
        num_heads: int,          # Attention heads
        window_size: int,        # Window size
        depth_pre: int,          # Pre-processing depth
        depth_mid: int,          # Main transformer depth
        depth_post: int,         # Post-processing depth
        mlp_ratio: float,        # MLP expansion ratio
        drop_path_rate: float,   # Stochastic depth
        input_height: int,       # Spatial height
        input_width: int,        # Spatial width
        use_checkpoint: bool,    # Gradient checkpointing
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - history frames
        Returns:
            (B, C, H, W) - predicted next frame
        """
        ...
```

**If your model uses different parameter names:**

Edit line 48 in `fuxi_train.py`:
```python
# Option 1: If your model is named FuXiModel
from model import FuXiModel as FuXi

# Option 2: If your model is in a package
from fuxi_model.core import FuXi

# Option 3: If parameters are named differently
# You'll need to create a wrapper - see below
```

**Model Wrapper Example** (if your interface differs):
```python
# Add to fuxi_train.py after imports
class FuXiWrapper(nn.Module):
    def __init__(self, num_variables, embed_dim, num_heads, window_size,
                 depth_pre, depth_mid, depth_post, mlp_ratio, drop_path_rate,
                 input_height, input_width, use_checkpoint):
        super().__init__()
        # Map to your model's parameter names
        from your_model import YourFuXiModel
        self.model = YourFuXiModel(
            channels=num_variables,
            dim=embed_dim,
            heads=num_heads,
            # ... map other parameters
        )
    
    def forward(self, x):
        return self.model(x)

# Then use FuXiWrapper instead of FuXi in the script
```

### 2. Update Data Paths

Edit `fuxi_sweep.sh` (line 18):
```bash
ZARR_STORE="/path/to/your/data.zarr"
```

Verify the zarr store exists:
```bash
ls -lh /path/to/your/data.zarr
```

### 3. Configure Conda Environment

Edit `fuxi_sweep.sh` (line 123) if your conda environment has a different name:
```bash
conda activate your_environment_name
```

Required packages:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xarray zarr numpy matplotlib
```

## 🚀 Usage

### Quick Start (Single Experiment)

```bash
# Single GPU
python fuxi_train.py \
    --zarr-store /path/to/data.zarr \
    --train-start 1979-01-01 \
    --train-end 2015-12-31 \
    --val-start 2016-01-01 \
    --val-end 2018-12-31 \
    --exp-name my_first_experiment

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 fuxi_train.py \
    --zarr-store /path/to/data.zarr \
    --train-start 1979-01-01 \
    --train-end 2015-12-31 \
    --batch-size 4 \
    --accum-steps 2
```

### Hyperparameter Sweep (Recommended)

```bash
# Make script executable
chmod +x fuxi_sweep.sh

# Submit all experiments
./fuxi_sweep.sh
```

This submits multiple SLURM jobs with different configurations:
- Paper baseline (256 dim, depth 2-12-2)
- Wider model (384 dim)
- Deeper model (depth 2-18-2)
- Large model (512 dim)
- Extra large (768 dim, A100 only)

### Monitor Training

```bash
# View queued/running jobs
squeue -u $USER

# Watch live training log
tail -f logs/pretrain_e256_*.out

# Check all experiment results
ls -lh Models/*/best.pt

# View metrics
cat Models/pretrain_e256_*/metrics.json
```

### Resume Failed Job

If a job crashes or times out:
```bash
python fuxi_train.py \
    --zarr-store /path/to/data.zarr \
    --resume Models/experiment_name/last.pt
```

The sweep script automatically resumes from `last.pt` if it exists.

## 🎛️ Key Hyperparameters

### Model Architecture

```bash
--embed-dim 256           # Embedding dimension (paper: 256)
--num-heads 8             # Attention heads (paper: 8)
--window-size 5           # Window size (paper: 5)
--depth-pre 2             # Pre-processing blocks (paper: 2)
--depth-mid 12            # Main transformer blocks (paper: 12)
--depth-post 2            # Post-processing blocks (paper: 2)
--drop-path-rate 0.2      # Stochastic depth (paper: 0.2)
```

### Training Configuration

```bash
--batch-size 4            # Batch size per GPU
--accum-steps 1           # Gradient accumulation (effective batch = batch × accum × GPUs)
--max-epochs 200          # Maximum epochs
--max-iters 40000         # Maximum iterations (paper value)
--patience 15             # Early stopping patience
```

### Optimizer (Paper Values)

```bash
--lr 2.5e-4               # Learning rate (paper: 2.5e-4 for pre-train)
--weight-decay 0.1        # Weight decay
--beta1 0.9               # Adam beta1
--beta2 0.95              # Adam beta2
--grad-clip 1.0           # Gradient clipping
```

### Memory Optimization

```bash
--use-checkpoint          # Enable gradient checkpointing (saves ~40% memory)
--accum-steps 4           # Use gradient accumulation instead of large batch
```

**Rule of thumb:**
- **24GB GPU:** batch_size=2, accum_steps=2, embed_dim≤256
- **40GB GPU:** batch_size=4, accum_steps=1, embed_dim≤384
- **80GB GPU:** batch_size=8, accum_steps=1, embed_dim≤768

## 📊 Output Files

Each experiment creates:

### Checkpoints
- `best.pt` - Best model based on validation loss
- `last.pt` - Latest checkpoint (for crash recovery)

### Metrics
```json
{
  "best_epoch": 42,
  "best_val_loss": 0.1234,
  "test_loss": 0.1245,
  "test_mae": 0.0567,
  "train_losses": [...],
  "val_losses": [...]
}
```

### Visualizations
- `loss_curve.png` - Training/validation loss over epochs
- `predictions_sample0.png` - Predicted vs ground truth maps

### Configuration
```json
{
  "zarr_store": "/path/to/data.zarr",
  "embed_dim": 256,
  "depth_mid": 12,
  "batch_size": 4,
  "lr": 0.00025,
  "num_parameters": 35421696,
  ...
}
```

## 🐛 Troubleshooting

### Import Error: Cannot find model

**Error:**
```
ImportError: cannot import name 'FuXi' from 'model'
```

**Solution:**
1. Check your model file name: `ls -l model.py`
2. Check your class name: `grep "class.*:" model.py`
3. Update import in `fuxi_train.py` line 48

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `--batch-size 2` or `--batch-size 1`
2. Use gradient accumulation: `--accum-steps 4`
3. Enable gradient checkpointing: `--use-checkpoint`
4. Reduce model size: `--embed-dim 192 --depth-mid 8`

### Zarr Store Not Found

**Error:**
```
ERROR: Zarr store not found: /path/to/data.zarr
```

**Solution:**
1. Check path: `ls -l /path/to/data.zarr`
2. Update `ZARR_STORE` in `fuxi_sweep.sh`
3. Ensure read permissions: `ls -ld /path/to/data.zarr`

### NCCL Errors (Multi-GPU)

**Error:**
```
NCCL error in: ...
```

**Solutions:**
1. Add to your environment:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1
   ```
2. Already included in sweep script
3. Try different NCCL backend: `export NCCL_SOCKET_IFNAME=eth0`

### Job Dies After 7 Days

**Solution:**
The sweep script saves `last.pt` every epoch. Jobs automatically resume from `last.pt` if it exists.

To manually resume:
```bash
sbatch -d afternotok:FAILED_JOB_ID resubmit_script.sh
```

## 📈 Expected Training Time

Approximate times on A100 80GB (paper configuration):

| GPUs | Batch/GPU | Accum | Effective Batch | Time/Epoch | Total Time (40k iters) |
|------|-----------|-------|-----------------|------------|------------------------|
| 1    | 4         | 1     | 4               | ~45 min    | ~312 hours             |
| 2    | 4         | 1     | 8               | ~23 min    | ~156 hours             |
| 4    | 4         | 1     | 16              | ~12 min    | ~78 hours              |
| 8    | 4         | 1     | 32              | ~6 min     | ~39 hours              |

**Note:** Larger effective batch sizes may require more iterations to converge.

## 🎯 Next Steps

After pre-training completes:

1. **Analyze results:**
   ```bash
   python -c "import json; print(json.load(open('Models/pretrain_*/metrics.json')))"
   ```

2. **Compare experiments:**
   ```bash
   for dir in Models/pretrain_*/; do
       echo "$dir: $(jq -r .test_loss $dir/metrics.json)"
   done | sort -t: -k2 -n
   ```

3. **Fine-tuning** (autoregressive, coming soon):
   - Will use best pre-trained checkpoint
   - Curriculum learning: 2→12 AR steps
   - Lower learning rate (1e-7)

4. **Inference** (coming soon):
   - Multi-step forecasting
   - Ensemble predictions
   - Metrics computation (RMSE, ACC)

## 📚 References

- FuXi Paper: [Nature Geoscience 2023]
- WeatherBench2: [arxiv.org/abs/2308.15560]
- ERA5 Data: [cds.climate.copernicus.eu]

## 💡 Tips

1. **Start small:** Test with 1 GPU, small model, few epochs first
2. **Monitor GPU usage:** `watch -n 1 nvidia-smi`
3. **Check data loading:** First epoch is slower (computes stats)
4. **Save frequently:** `last.pt` saves every epoch (crash recovery)
5. **Use tensorboard:** Add `--tensorboard` flag (coming soon)

## ✅ Checklist

Before running the sweep:

- [ ] Updated `ZARR_STORE` path in `fuxi_sweep.sh`
- [ ] Verified model import in `fuxi_train.py`
- [ ] Activated correct conda environment
- [ ] Created `logs/` directory: `mkdir -p logs`
- [ ] Checked GPU availability: `nvidia-smi`
- [ ] Tested single experiment first
- [ ] Reviewed SLURM partition names

Ready to train! 🚀
