# Medical Image Classification with Simple CNN

This project implements a basic CNN to classify medical images (e.g., skin lesions) into benign or malignant categories using the HAM10000 dataset. currently the model performs at an accuracy of 0.77

<p align="center">
  <img src="https://github.com/dhana2403/med-cnn-classifier/blob/main/cnn_archi.png" width="900"/>
</p>

## Project Structure
- `models/`: CNN architecture
- `data/`: Dataset management
- `prepare_data.py/`: Data preparation(train, test data split)
- `train.py`: Training script
- `test.py`: Prediction on test data
- `notebooks/`: EDA

<p align="center">
  <img src="https://github.com/dhana2403/med-cnn-classifier/blob/main/workflow.png" width="900"/>
</p>

## Results
<p align="right">
  <img src="https://github.com/dhana2403/med-cnn-classifier/blob/main/roc_curve.png" width="500"/>
</p>
<p align="left">
  <img src="https://github.com/dhana2403/med-cnn-classifier/blob/main/precision_recall_curve.png" width="500"/>
</p>

## To Run:
```bash
pip install -r requirements.txt
python data/download_data.py
python train.py
python test.py
```
