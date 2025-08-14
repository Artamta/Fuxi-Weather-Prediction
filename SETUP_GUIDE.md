# Multi-Device Development Setup Guide

## Weather Forecast Project - Local MacBook + Cluster Computer

### 🎯 **Goal**: Seamless development between your MacBook and cluster computer

---

## 📋 **Current Setup Status**

✅ Git repository initialized  
✅ Connected to GitHub: `https://github.com/Artamta/Fuxi-Weather-Prediction.git`  
✅ SSH key generated for GitHub authentication  
✅ `.gitignore` configured for Python/ML projects  
✅ SSH config set up for cluster: `raj.ayush@192.168.10.3`

**Cluster Details:**

- Host: `weather-cluster` (192.168.10.3)
- User: `raj.ayush`
- Shared system: Git config set per-project only

---

## 🖥️ **Setup on Cluster Computer**

### 1. Clone Repository on Cluster

```bash
# SSH into your cluster
ssh raj.ayush@192.168.10.3

# Clone the repository (use HTTPS since SSH keys might not be set up)
git clone https://github.com/Artamta/Fuxi-Weather-Prediction.git
cd Fuxi-Weather-Prediction

# Configure Git for THIS PROJECT ONLY (not global on shared cluster)
git config user.name "Artamta"
git config user.email "artamta47@gmail.com"

# Verify your configuration
git config user.name
git config user.email
```

### 2. Git Configuration for Shared Cluster

```bash
# IMPORTANT: On shared clusters, DON'T use --global
# This sets config only for this project/repository

# Inside your project directory:
cd Fuxi-Weather-Prediction

# Set your identity for this project only
git config user.name "Artamta"
git config user.email "artamta47@gmail.com"

# Optional: Set up credential helper for this project
git config credential.helper store

# Check your settings
git config --list --local
```

---

## 🔄 **Daily Workflow**

### **Working on MacBook (Local Development)**

```bash
# Start work session
git pull origin main

# Make changes, test locally
# ... your development work ...

# Save progress to GitHub
git add .
git commit -m "Describe your changes"
git push origin main
```

### **Working on Cluster (Heavy Computation)**

```bash
# Start work session
git pull origin main

# Run heavy computations, training, etc.
# ... your cluster work ...

# Save results back to GitHub
git add .
git commit -m "Add training results/models"
git push origin main
```

---

## 🌐 **Remote Development Options**

### **Option 1: VS Code Remote SSH** (Recommended)

1. Install Remote-SSH extension in VS Code
2. Connect directly to cluster from VS Code
3. Full VS Code experience on cluster

```bash
# Add to ~/.ssh/config on MacBook
Host weather-cluster
    HostName 192.168.10.3
    User raj.ayush
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

### **Option 2: GitHub Codespaces**

- Develop in cloud environment
- Access from any device
- Pre-configured development environment

### **Option 3: File Synchronization**

```bash
# Sync files using rsync
rsync -avz --delete /Users/ayush/Desktop/weather_forcast/ username@cluster:/path/to/project/

# Or use git as primary sync method (recommended)
```

---

## 🐍 **Python Environment Management**

### **Conda Environment Setup (Recommended)**

#### On MacBook:

```bash
# Create and activate environment
conda create -n weather_forecast python=3.9 -y
conda activate weather_forecast

# Install packages via conda
conda install -c conda-forge numpy pandas scipy matplotlib seaborn plotly \
    jupyter notebook ipykernel xarray netcdf4 h5py tqdm pyyaml -y

# Install remaining packages via pip
pip install torch torchvision scikit-learn python-dotenv black flake8 pytest

# Export environment for cluster
conda env export > environment.yml
pip freeze > requirements.txt
```

#### On Cluster:

```bash
# Option 1: Use conda environment file
conda env create -f environment.yml
conda activate weather_forecast

# Option 2: Use pip requirements (if conda not available)
pip install -r requirements.txt

# Option 3: Manual conda setup
conda create -n weather_forecast python=3.9 -y
conda activate weather_forecast
pip install -r requirements.txt
```

### **Environment Activation:**

```bash
# Always activate before working
conda activate weather_forecast

# Verify environment
python -c "import torch, numpy, xarray; print('Environment ready!')"
```

---

## 📁 **Project Structure Recommendations**

```
weather_forcast/
├── data/              # Raw data (add to .gitignore if large)
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code
├── models/           # Trained models (consider git-lfs for large files)
├── results/          # Experiment results
├── configs/          # Configuration files
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── .gitignore        # Git ignore rules
```

---

## 🚀 **Best Practices**

### **Before Starting Work:**

1. Always run `git pull` first
2. Check for conflicts
3. Ensure environment is up to date

### **During Development:**

1. Commit frequently with descriptive messages
2. Push major milestones
3. Use branches for experimental features

### **Large Files Management:**

```bash
# For large datasets or models, consider Git LFS
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.model"
```

### **Cluster-Specific Considerations:**

- Use screen/tmux for long-running processes
- Set up proper logging
- Consider using job schedulers (SLURM, PBS)

---

## 🔧 **VS Code Remote SSH Setup**

### 1. Configure SSH connection:

```bash
# On MacBook, edit ~/.ssh/config
Host weather-cluster
    HostName your-cluster-address
    User your-username
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

### 2. Connect from VS Code:

- Open Command Palette (Cmd+Shift+P)
- Type "Remote-SSH: Connect to Host"
- Select your cluster
- Open the project folder remotely

---

## 📊 **Example Development Scenarios**

### **Scenario 1: Data Exploration on MacBook**

```bash
# Local work
git pull
# Work in Jupyter notebooks locally
git add notebooks/exploration.ipynb
git commit -m "Add initial data exploration"
git push
```

### **Scenario 2: Model Training on Cluster**

```bash
# On cluster
git pull
# Run training scripts
python src/train_model.py
git add models/trained_model.pkl results/training_metrics.json
git commit -m "Train weather prediction model"
git push
```

### **Scenario 3: Results Analysis on MacBook**

```bash
# Back on MacBook
git pull  # Get the trained model
# Analyze results in notebooks
git add notebooks/results_analysis.ipynb
git commit -m "Analyze model performance"
git push
```

---

## 🆘 **Troubleshooting**

### **Common Issues:**

1. **Merge conflicts**: Use `git pull --rebase` or resolve manually
2. **Large files**: Consider Git LFS or exclude from repo
3. **SSH connection issues**: Check network, keys, and permissions
4. **Environment differences**: Keep requirements.txt updated

### **Emergency Commands:**

```bash
# Force sync (dangerous - only if you're sure)
git reset --hard origin/main

# Stash changes temporarily
git stash
git pull
git stash pop

# Check what's different
git status
git diff
```

---

## 📞 **Quick Reference**

| Task                 | Command                                            |
| -------------------- | -------------------------------------------------- |
| Sync with GitHub     | `git pull && git push`                             |
| Save progress        | `git add . && git commit -m "message" && git push` |
| Check status         | `git status`                                       |
| View history         | `git log --oneline`                                |
| Connect to cluster   | `ssh weather-cluster`                              |
| Start remote VS Code | Cmd+Shift+P → "Remote-SSH: Connect"                |

---

**Next Steps:**

1. Set up cluster access details
2. Install VS Code Remote extensions
3. Configure SSH connection
4. Test the workflow end-to-end
