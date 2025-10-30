srun --partition=gpu --gres=gpu:1 --cpus-per-task=32 --mem=64 --pty bash
salloc --partition=iiser --gres=gpu:1 --cpus-per-task=32 --mem=256G