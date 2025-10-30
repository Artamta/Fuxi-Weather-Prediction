"""
Commands:

srun --partition=gpu --gres=gpu:1 --cpus-per-task=32 --mem=64 --pty bash
salloc --partition=iiser --gres=gpu:1 --cpus-per-task=32 --mem=256G
salloc --partition=iiser --gres=gpu:1 --cpus-per-task=32 --mem=256G --job-name=Fuxi_run

salloc --partition=GPU-AI_prio --gres=gpu:1 --cpus-per-task=32 --mem=512G --job-name=Fuxi_Train

jupyter lab --no-browser --port=8888

ssh -L 8888:localhost:8888 -J raj.ayush@192.168.10.3 raj.ayush@cn1


"""