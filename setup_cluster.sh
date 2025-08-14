#!/bin/bash
# Cluster Setup Script for Weather Forecasting Project
# Run this script on your cluster after cloning the repository

echo "🌦️  Setting up Weather Forecasting Environment on Cluster"
echo "=========================================================="

# Check current directory
if [ ! -f "requirements_clean.txt" ] || [ ! -f "environment_cluster.yml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected files: requirements_clean.txt, environment_cluster.yml"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo "📋 Contents:"
ls -la

# Configure Git for this project
echo "🔧 Configuring Git for this project..."
git config user.name "Artamta"
git config user.email "artamta47@gmail.com"
echo "✅ Git configured"

# Check what's available
echo "🔍 Checking available tools..."
if command -v conda &> /dev/null; then
    echo "✅ Conda found: $(conda --version)"
    USE_CONDA=true
elif command -v python3 &> /dev/null; then
    echo "✅ Python3 found: $(python3 --version)"
    USE_CONDA=false
else
    echo "❌ Neither conda nor python3 found!"
    exit 1
fi

# Setup environment
if [ "$USE_CONDA" = true ]; then
    echo "🐍 Setting up Conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "weather_forecast"; then
        echo "⚠️  Environment 'weather_forecast' already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n weather_forecast -y
        else
            echo "ℹ️  Using existing environment"
        fi
    fi
    
    # Create environment
    if ! conda env list | grep -q "weather_forecast"; then
        echo "📦 Creating conda environment from environment_cluster.yml..."
        conda env create -f environment_cluster.yml
    fi
    
    echo "🔄 Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate weather_forecast
    
    echo "✅ Conda environment activated"
    
else
    echo "🐍 Setting up Python virtual environment..."
    
    # Create virtual environment
    if [ -d "weather_forecast_env" ]; then
        echo "⚠️  Virtual environment already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf weather_forecast_env
        fi
    fi
    
    if [ ! -d "weather_forecast_env" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv weather_forecast_env
    fi
    
    echo "🔄 Activating virtual environment..."
    source weather_forecast_env/bin/activate
    
    echo "📦 Installing packages..."
    pip install --upgrade pip
    pip install -r requirements_clean.txt
    
    echo "✅ Virtual environment activated and packages installed"
fi

# Test the environment
echo "🧪 Testing environment..."
python test_environment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📝 To activate this environment in future sessions:"
    if [ "$USE_CONDA" = true ]; then
        echo "   conda activate weather_forecast"
    else
        echo "   source weather_forecast_env/bin/activate"
    fi
    echo ""
    echo "🚀 You're ready to start weather forecasting!"
    echo "   - Run Jupyter: jupyter notebook"
    echo "   - Edit code in src/"
    echo "   - Remember to git pull/push to sync with your MacBook"
else
    echo ""
    echo "⚠️  Some packages may not have installed correctly"
    echo "   Check the error messages above and install missing packages manually"
fi
