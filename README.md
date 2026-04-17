# Breast Cancer Classification with ANN: Early Stopping & Dropout Analysis

## Overview
This project implements an Artificial Neural Network (ANN) for breast cancer classification, analyzing the effects of:
- **Early Stopping**: Prevents overfitting by stopping training when validation performance degrades
- **Dropout Layers**: Randomly drops neurons during training to improve generalization

## Dataset
Uses the Wisconsin Breast Cancer Dataset from scikit-learn with 569 samples and 30 features.

## Key Features
- Comparative analysis of model performance with/without early stopping and dropout
- Visualization of training dynamics and decision boundaries
- Performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Comprehensive model evaluation and comparison

## Installation

```bash
git clone https://github.com/yourusername/breast-cancer-ann-analysis.git
cd breast-cancer-ann-analysis
pip install -r requirements.txt
