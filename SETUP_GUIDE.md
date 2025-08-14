# Multi-Device Development Setup Guide
## Weather Forecast Project - Local MacBook + Cluster Computer

### 🎯 **Goal**: Seamless development between your MacBook and cluster computer

---

## 📋 **Current Setup Status**
✅ Git repository initialized  
✅ Connected to GitHub: `https://github.com/Artamta/Fuxi-Weather-Prediction.git`  
✅ SSH key generated for GitHub authentication  
✅ `.gitignore` configured for Python/ML projects  

---

## 🖥️ **Setup on Cluster Computer**

### 1. Clone Repository on Cluster
```bash
# SSH into your cluster
ssh username@cluster-address

# Clone the repository
git clone git@github.com:Artamta/Fuxi-Weather-Prediction.git
cd Fuxi-Weather-Prediction

# Configure Git (if not already done)
git config --global user.name "Artamta"
git config --global user.email "artamta47@gmail.com"
```

### 2. Add SSH Key to Cluster (if needed)
```bash
# On cluster, generate SSH key
ssh-keygen -t ed25519 -C "artamta47@gmail.com"

# Display public key to add to GitHub
cat ~/.ssh/id_ed25519.pub
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
Host cluster
    HostName your-cluster-ip
    User your-username
    IdentityFile ~/.ssh/id_ed25519
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

### **Keep environments consistent:**
```bash
# Export environment from MacBook
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add requirements.txt"
git push

# On cluster, install same environment
git pull
pip install -r requirements.txt

# Or use conda
conda env export > environment.yml
# On cluster: conda env create -f environment.yml
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

| Task | Command |
|------|---------|
| Sync with GitHub | `git pull && git push` |
| Save progress | `git add . && git commit -m "message" && git push` |
| Check status | `git status` |
| View history | `git log --oneline` |
| Connect to cluster | `ssh weather-cluster` |
| Start remote VS Code | Cmd+Shift+P → "Remote-SSH: Connect" |

---

**Next Steps:**
1. Set up cluster access details
2. Install VS Code Remote extensions
3. Configure SSH connection
4. Test the workflow end-to-end
