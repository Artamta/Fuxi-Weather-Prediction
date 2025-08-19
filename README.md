# Fuxi Weather Prediction Project

This project implements weather forecasting using the Fuxi model architecture.

## Project Structure

```
weather_forecast/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml
│   ├── data_config.yaml
│   └── train_config.yaml
├── data/
│   ├── download_era5.py
│   ├── preprocess.py
│   └── dataset.py
├── fuxi/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fuxi_base.py
│   │   ├── u_transformer.py
│   │   └── cube_embedding.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── cascade_trainer.py
│   │   └── losses.py
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── ensemble_generator.py
│   │   └── perturbations.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── visualization.py
│       └── io_utils.py
├── scripts/
│   ├── train_single.py
│   ├── train_cascade.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── results_visualization.ipynb
└── tests/
    ├── test_models.py
    ├── test_data.py
    └── test_training.py       # This file
```

## Setup

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start developing!

### Cluster Development

1. Clone the repository on cluster
2. Set up the same environment
3. Use git to sync changes

## Workflow

- Develop and explore on local MacBook
- Train large models on cluster
- Use Git to keep everything in sync

See `SETUP_GUIDE.md` for detailed multi-device workflow instructions.

## Contributing

1. Pull latest changes: `git pull`
2. Make your changes
3. Commit and push: `git add . && git commit -m "message" && git push`
