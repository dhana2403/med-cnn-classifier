# Medical Image Classification with Simple CNN

This project implements a basic CNN to classify medical images (e.g., skin lesions) into benign or malignant categories using the HAM10000 dataset.

## Project Structure
- `models/`: CNN architecture
- `data/`: Dataset management
- `train.py`: Training script
- `eval.py`: Prediction on test data
- `notebooks/`: EDA 
## Goals
- Build a reproducible deep learning pipeline
- Practice clean, modular PyTorch code
- Evaluate real-world biomedical application

## To Run:
```bash
pip install -r requirements.txt
python data/download_data.py
python train.py
python test.py
```

## TODO:
- [ ] Improve model architecture
- [ ] Add Docker container
- [ ] Add metrics and visualizations
"""
