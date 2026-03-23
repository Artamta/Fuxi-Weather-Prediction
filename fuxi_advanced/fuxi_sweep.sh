#!/bin/bash
# =============================================================================
# FuXi Pre-training Hyperparameter Sweep
#
# Submits multiple SLURM jobs with different model configurations.
# Reads directly from Zarr store — zero data copying required.
#
# Usage:  bash fuxi_sweep.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/Models"

# =============================================================================
# Data Configuration
# =============================================================================

# Path to your Zarr store
ZARR_STORE="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# Time ranges (following paper split)
TRAIN_START="1979-01-01"
TRAIN_END="2015-12-31"
VAL_START="2016-01-01"
VAL_END="2018-12-31"
TEST_START="2019-01-01"
TEST_END="2020-12-31"

# =============================================================================
# Verify Zarr Store Exists
# =============================================================================

if [ ! -d "${ZARR_STORE}" ]; then
    echo "ERROR: Zarr store not found: ${ZARR_STORE}"
    echo "Please update ZARR_STORE path in this script."
    exit 1
fi

# =============================================================================
# Experiment Configurations
# =============================================================================

# Format: "EMBED_DIM|NUM_HEADS|WINDOW_SIZE|DEPTH_PRE|DEPTH_MID|DEPTH_POST|BATCH_SIZE|LR|DROP_PATH|ACCUM_STEPS|MAX_ITERS"
#
# Paper configuration:
#   - embed_dim: 256
#   - num_heads: 8
#   - window_size: 5
#   - depth: pre=2, mid=12, post=2
#   - lr: 2.5e-4
#   - max_iters: 40000

EXPERIMENTS=(
    # ---- Paper baseline (small) ----
    "256|8|5|2|12|2|4|2.5e-4|0.2|1|40000"
    
    # ---- Wider model ----
    "384|12|5|2|12|2|4|2.5e-4|0.2|1|40000"
    
    # ---- Deeper mid-section ----
    "256|8|5|2|18|2|4|2.5e-4|0.25|1|40000"
    
    # ---- Large model (reduced batch for memory) ----
    "512|16|5|2|16|2|2|2.0e-4|0.25|2|40000"
    
    # ---- Extra large (A100 80GB only) ----
    "768|16|7|3|20|3|1|1.5e-4|0.3|4|40000"
)

# =============================================================================
# Common Training Settings
# =============================================================================

MAX_EPOCHS=200        # Maximum epochs (usually hits max_iters first)
PATIENCE=15           # Early stopping patience
GRAD_CLIP=1.0         # Gradient clipping
WEIGHT_DECAY=0.1      # AdamW weight decay
BETA1=0.9             # Adam beta1
BETA2=0.95            # Adam beta2
NUM_WORKERS=4         # DataLoader workers
USE_CHECKPOINT=""     # Add "--use-checkpoint" to enable gradient checkpointing

# =============================================================================
# SLURM Configuration
# =============================================================================

PARTITION="gpu"              # Default partition
PARTITION_LARGE="GPU-AI"     # For large models (embed >= 512)
TIME_LIMIT="7-00:00:00"      # 7 days
EXCLUDE_NODES="cn3,cn15"     # Problematic nodes to exclude

# =============================================================================
# Logging Setup
# =============================================================================

mkdir -p logs
mkdir -p "${RUNS_DIR}"

echo "============================================================"
echo "  FuXi Pre-training Hyperparameter Sweep"
echo "  ${#EXPERIMENTS[@]} experiments to submit"
echo "============================================================"
echo ""
echo "Zarr store: ${ZARR_STORE}"
echo "Train     : ${TRAIN_START} → ${TRAIN_END}"
echo "Val       : ${VAL_START} → ${VAL_END}"
echo "Test      : ${TEST_START} → ${TEST_END}"
echo ""
echo "============================================================"
echo ""

# =============================================================================
# Submit Jobs
# =============================================================================

SUBMITTED=0
SKIPPED=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EMBED_DIM NUM_HEADS WINDOW_SIZE DEPTH_PRE DEPTH_MID DEPTH_POST \
                    BATCH_SIZE LR DROP_PATH ACCUM_STEPS MAX_ITERS <<< "$exp"
    
    # Create experiment name
    EXP_NAME="pretrain_e${EMBED_DIM}_h${NUM_HEADS}_w${WINDOW_SIZE}_d${DEPTH_PRE}-${DEPTH_MID}-${DEPTH_POST}_bs${BATCH_SIZE}"
    
    # Check if already trained (has best.pt)
    if [ -f "${RUNS_DIR}/${EXP_NAME}/best.pt" ]; then
        echo "SKIP: ${EXP_NAME} (already has best.pt)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Check for resume
    RESUME_FLAG=""
    LAST_CKPT="${RUNS_DIR}/${EXP_NAME}/last.pt"
    if [ -f "${LAST_CKPT}" ]; then
        RESUME_FLAG="--resume ${LAST_CKPT}"
        echo "RESUME: ${EXP_NAME} (from last.pt)"
    fi
    
    # Determine GPU requirements based on model size
    if [ "${EMBED_DIM}" -ge 512 ]; then
        GPUS=4
        MEM="160G"
        CPUS=16
        PARTITION_USE="${PARTITION_LARGE}"
    elif [ "${EMBED_DIM}" -ge 384 ]; then
        GPUS=2
        MEM="96G"
        CPUS=12
        PARTITION_USE="${PARTITION}"
    else
        GPUS=1
        MEM="48G"
        CPUS=8
        PARTITION_USE="${PARTITION}"
    fi
    
    # Calculate effective batch size
    EFFECTIVE_BS=$((BATCH_SIZE * ACCUM_STEPS * GPUS))
    
    echo "SUBMIT: ${EXP_NAME}"
    echo "  Config: embed=${EMBED_DIM}, heads=${NUM_HEADS}, depths=${DEPTH_PRE}-${DEPTH_MID}-${DEPTH_POST}"
    echo "  Batch : ${BATCH_SIZE}/GPU × ${ACCUM_STEPS} accum × ${GPUS} GPUs = ${EFFECTIVE_BS} effective"
    echo "  GPUs  : ${GPUS} on ${PARTITION_USE}"
    
    # Submit SLURM job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --partition=${PARTITION_USE}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclude=${EXCLUDE_NODES}
#SBATCH --output=logs/${EXP_NAME}_%j.out
#SBATCH --error=logs/${EXP_NAME}_%j.err

set -euo pipefail

# Activate conda environment
source /apps/compilers/anaconda3-2023.3/etc/profile.d/conda.sh
conda activate weather_forecast

cd ${SCRIPT_DIR}

# Detect GPUs
NGPUS=\$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "============================================================"
echo "Job ID     : \${SLURM_JOB_ID}"
echo "Experiment : ${EXP_NAME}"
echo "Node       : \$(hostname)"
echo "GPUs       : \${NGPUS}"
echo "============================================================"
nvidia-smi
echo "============================================================"
echo ""

# Environment variables for multi-GPU training
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues
MASTER_PORT=\$((29500 + RANDOM % 1000))

# Launch training
if [ "\${NGPUS}" -gt 1 ]; then
    echo "Launching with torchrun (DDP mode)..."
    torchrun \\
        --nproc_per_node=\${NGPUS} \\
        --master_port=\${MASTER_PORT} \\
        fuxi_train.py \\
        --zarr-store "${ZARR_STORE}" \\
        --train-start "${TRAIN_START}" \\
        --train-end "${TRAIN_END}" \\
        --val-start "${VAL_START}" \\
        --val-end "${VAL_END}" \\
        --test-start "${TEST_START}" \\
        --test-end "${TEST_END}" \\
        --exp-name "${EXP_NAME}" \\
        --runs-dir "${RUNS_DIR}" \\
        --embed-dim ${EMBED_DIM} \\
        --num-heads ${NUM_HEADS} \\
        --window-size ${WINDOW_SIZE} \\
        --depth-pre ${DEPTH_PRE} \\
        --depth-mid ${DEPTH_MID} \\
        --depth-post ${DEPTH_POST} \\
        --batch-size ${BATCH_SIZE} \\
        --accum-steps ${ACCUM_STEPS} \\
        --max-epochs ${MAX_EPOCHS} \\
        --max-iters ${MAX_ITERS} \\
        --patience ${PATIENCE} \\
        --lr ${LR} \\
        --weight-decay ${WEIGHT_DECAY} \\
        --beta1 ${BETA1} \\
        --beta2 ${BETA2} \\
        --grad-clip ${GRAD_CLIP} \\
        --drop-path-rate ${DROP_PATH} \\
        --num-workers ${NUM_WORKERS} \\
        ${USE_CHECKPOINT} \\
        ${RESUME_FLAG}
else
    echo "Launching single-GPU training..."
    python3 fuxi_train.py \\
        --zarr-store "${ZARR_STORE}" \\
        --train-start "${TRAIN_START}" \\
        --train-end "${TRAIN_END}" \\
        --val-start "${VAL_START}" \\
        --val-end "${VAL_END}" \\
        --test-start "${TEST_START}" \\
        --test-end "${TEST_END}" \\
        --exp-name "${EXP_NAME}" \\
        --runs-dir "${RUNS_DIR}" \\
        --embed-dim ${EMBED_DIM} \\
        --num-heads ${NUM_HEADS} \\
        --window-size ${WINDOW_SIZE} \\
        --depth-pre ${DEPTH_PRE} \\
        --depth-mid ${DEPTH_MID} \\
        --depth-post ${DEPTH_POST} \\
        --batch-size ${BATCH_SIZE} \\
        --accum-steps ${ACCUM_STEPS} \\
        --max-epochs ${MAX_EPOCHS} \\
        --max-iters ${MAX_ITERS} \\
        --patience ${PATIENCE} \\
        --lr ${LR} \\
        --weight-decay ${WEIGHT_DECAY} \\
        --beta1 ${BETA1} \\
        --beta2 ${BETA2} \\
        --grad-clip ${GRAD_CLIP} \\
        --drop-path-rate ${DROP_PATH} \\
        --num-workers ${NUM_WORKERS} \\
        ${USE_CHECKPOINT} \\
        ${RESUME_FLAG}
fi

EXIT_CODE=\$?

echo ""
echo "============================================================"
echo "Job \${SLURM_JOB_ID} finished at \$(date)"
echo "Exit code: \${EXIT_CODE}"
echo "============================================================"

exit \${EXIT_CODE}
EOF

    SUBMITTED=$((SUBMITTED + 1))
    sleep 0.5  # Small delay between submissions
    echo ""
done

echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "Submitted: ${SUBMITTED}"
echo "Skipped  : ${SKIPPED}"
echo "Total    : ${#EXPERIMENTS[@]}"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "View logs  : tail -f logs/pretrain_*.out"
echo "Results    : ls -lh ${RUNS_DIR}/*/best.pt"
echo "============================================================"
