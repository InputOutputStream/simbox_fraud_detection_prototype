# Fraud Detection System

A comprehensive machine learning system for detecting fraudulent telecommunications users based on Call Detail Records (CDR) analysis using deep learning and ensemble methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Workflow](#pipeline-workflow)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Results](#results)

## 🎯 Overview

This system implements a sophisticated fraud detection pipeline that:
- Processes raw CDR data from telecommunication operators
- Applies advanced feature engineering with PCA
- Trains deep learning models (MLP) for binary classification
- Supports ensemble learning with multiple stacking methods
- Handles imbalanced datasets using SMOTE and undersampling
- Provides comprehensive visualization and evaluation tools

## ✨ Features

### Core Capabilities
- **Multi-stage Data Processing**: Text-to-CSV conversion, categorical discretization, train/val/test splitting
- **Advanced Preprocessing**: PCA-based dimensionality reduction with component interpretation
- **Imbalanced Data Handling**: Iterative SMOTE + Random Undersampling
- **Deep Learning**: Multi-layer perceptron with dropout regularization
- **Ensemble Methods**: Support for averaging, weighted, and meta-learning stacking
- **Robustness Testing**: Gaussian noise injection for model stability evaluation
- **Rich Visualizations**: Training curves, confusion matrices, ROC curves, PCA analysis

### Model Types
1. **Single Models**: Traffic, Mobility, Social, All-Naive pattern detection
2. **Ensemble System**: Combines multiple specialized models for improved accuracy

## 📁 Project Structure

```
fraud-detection/
├── model.py                    # Core MLP fraud detection system
├── ensemble.py                 # Ensemble learning implementation
├── preprocessor.py             # Data preprocessing with PCA
├── config.py                   # Centralized configuration
├── txt2csv.py                  # Convert raw .txt CDR to .csv
├── categorize_data.py          # Flag fraudulent users in datasets
├── train_test_val_split.py     # Split data with stratification
├── Tests/
│   ├── 5sims.py               # Test with 5 simulations
│   ├── 50sims.py              # Test with 50 simulations
│   ├── 100sims.py             # Test with 100 simulations
│   ├── 150sims.py             # Test with 150 simulations
│   └── 200sims.py             # Test with 200 simulations
├── models/                     # Saved model weights
├── plots/                      # Generated visualizations
└── TestCDR/                    # CDR datasets
    ├── advanced_traffic/
    ├── advanced_mobility/
    ├── advanced_social/
    └── all_naive/
```

## 🚀 Installation

### Requirements
```bash
# Core dependencies
pip install torch numpy pandas scikit-learn
pip install imbalanced-learn matplotlib seaborn
```

### Python Version
- Python 3.8+

## ⚡ Quick Start

### 1. Single Model Training

```python
from model import FraudDetectionSystem

# Initialize system
fraud_detector = FraudDetectionSystem(
    model_params={'hidden_size': 64, 'dropout_rate': 0.2},
    random_state=42
)

# Load and preprocess data
X, y, feature_names = fraud_detector.load_and_preprocess_data(
    'TestCDR/all_naive/all_naive_12%_5sims/Op_1_CDRTrace_flagged.csv',
    use_pca=True
)

# Prepare training data
data_tensors = fraud_detector.prepare_training_data(X, y, apply_sampling=True)

# Train model
history = fraud_detector.train(
    data_tensors, 
    num_epochs=2000, 
    lr=0.001, 
    timer=100
)

# Evaluate
results = fraud_detector.evaluate(
    data_tensors['X_test'], 
    data_tensors['y_test']
)

# Save model
fraud_detector.save_model('models/fraud_detector.pth')
```

### 2. Ensemble Learning

```python
from ensemble import EnsembleSystem

# Initialize ensemble with meta-learning
ensemble = EnsembleSystem(stacking_method="meta")

# Define data paths for each specialized model
train_data_paths = {
    "traffic": "TestCDR/advanced_traffic/traffic_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "mobility": "TestCDR/advanced_mobility/mobility_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "social": "TestCDR/advanced_social/social_12%_50sims/Op_1_CDRTrace_flagged.csv",
    "all_naive": "TestCDR/all_naive/all_naive_12%_50sims/Op_1_CDRTrace_flagged.csv"
}

# Train all base models
ensemble.fit_base_models(train_data_paths, num_epochs=2000, timer=100)

# Load validation data for meta-learner
X_val, y_val = ensemble.load_validation_data(val_data_paths["all_naive"])

# Train meta-learner
ensemble.fit(X_val, y_val)

# Evaluate on test set
X_test, y_test = ensemble.load_test_data(test_data_paths["all_naive"])
results = ensemble.evaluate(X_test, y_test)

# Visualize ensemble performance
ensemble.visualize_ensemble_performance(X_test, y_test)

# Save complete ensemble
ensemble.save_ensemble()
```

## 🔄 Pipeline Workflow

### Stage 1: Data Preparation
```bash
# 1. Convert raw text files to CSV
python txt2csv.py

# 2. Mark fraudulent users
python categorize_data.py

# 3. Split into train/val/test sets with stratification
python train_test_val_split.py
```

### Stage 2: Model Training
```bash
# Run tests with different simulation sizes
python Tests/5sims.py      # Quick testing
python Tests/50sims.py     # Standard evaluation
python Tests/100sims.py    # Extended evaluation
```

### Stage 3: Evaluation & Visualization
The system automatically generates:
- Training loss curves
- Validation accuracy plots
- Confusion matrices
- ROC curves
- PCA component analysis
- Feature importance visualizations

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model Architecture
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.2
NUM_EPOCHS = 2000
LEARNING_RATE = 0.001

# Data Processing
USE_PCA = True
PCA_VARIANCE_THRESHOLD = 0.95
APPLY_SAMPLING = True
SAMPLING_ITERATIONS = 3

# Data Splits
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature Columns
FEATURE_COLUMNS = [
    'size', 'src_number', 'dst_number', 'duration', 
    'hour', 'CDR_type', 'dst_imei', 'src_imei', 
    'minute', 'second', 'current_src_cellId'
]
```

## 📊 Usage Examples

### Predict on New Data
```python
# Load trained model
detector = FraudDetectionSystem()
detector.load_model('models/fraud_detector.pth')

# Predict on new CDR data
results = detector.predict_new_data(
    'TestCDR/new_data/Op_1_CDRTrace_test_split.csv',
    save_plots=True
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(results['classification_report'])
```

### Robustness Testing
```python
# Test model stability with noise
robustness_results = fraud_detector.test_robustness(
    X, y, 
    noise_levels=[0.1, 0.2, 0.5]
)

# Visualize robustness
fraud_detector.visualize_robustness_test(robustness_results)
```

### PCA Component Analysis
```python
# Get feature analysis
analysis = fraud_detector.get_feature_analysis()
print(f"Using {analysis['n_components']} principal components")
print(f"Explained variance: {analysis['explained_variance']:.2%}")

# View component interpretations
for pc, features in analysis['component_explanations'].items():
    print(f"{pc}: {', '.join(features)}")
```

### Ensemble Stacking Methods

```python
# Method 1: Simple Averaging
ensemble = EnsembleSystem(stacking_method="average")

# Method 2: Weighted Averaging (based on validation performance)
ensemble = EnsembleSystem(stacking_method="weighted")

# Method 3: Meta-Learning (neural network combines predictions)
ensemble = EnsembleSystem(stacking_method="meta")
```

## 🏗️ Model Architecture

### Base MLP Model
```
Input Layer (n_features after PCA)
    ↓
Hidden Layer 1 (64 neurons) + ReLU + Dropout(0.2)
    ↓
Hidden Layer 2 (32 neurons) + ReLU + Dropout(0.2)
    ↓
Output Layer (2 classes: Legitimate/Fraudulent)
```

### Ensemble Architecture
```
Base Models:
├── Traffic Pattern Model
├── Mobility Pattern Model
├── Social Network Model
└── All-Naive Pattern Model
    ↓
Meta-Learner (Neural Network)
    ↓
Final Prediction
```

## 📈 Results

### Performance Metrics
- **Accuracy**: Typically 90-95% on test sets
- **Precision/Recall**: Balanced detection of fraud cases
- **AUC-ROC**: Strong discriminative ability
- **Robustness**: Maintains performance under noise

### Visualization Outputs
The system generates comprehensive plots in the `plots/` directory:
- `training_history.png`: Loss and accuracy curves
- `performance.png`: Overall model performance
- `confusion_matrix.png`: Classification breakdown
- `roc_curve.png`: ROC analysis
- `pca_analysis.png`: Component interpretation

## 🔬 Advanced Features

### Adaptive Learning (Future Development)
```python
# Files prepared for future implementation:
# - adaptive_fraud_detector.py: Incremental learning
# - continouos_learning.py: Real-time updates
```

### Custom Model Configurations
```python
# Use predefined architectures
from config import ARCHITECTURES

# Small model for fast training
fraud_detector = FraudDetectionSystem(
    model_params=ARCHITECTURES["small"]
)

# Large model for maximum accuracy
fraud_detector = FraudDetectionSystem(
    model_params=ARCHITECTURES["large"]
)
```

## 📝 Notes

- **Data Format**: Expects CDR files with columns defined in `config.py`
- **GPU Support**: Automatically uses CUDA if available
- **Memory**: PCA significantly reduces memory footprint
- **Reproducibility**: Set `RANDOM_SEED` in config for consistent results

## 🤝 Contributing

When adding new features:
1. Update `config.py` with new parameters
2. Add tests in the `Tests/` directory
3. Update this README with usage examples
4. Ensure backward compatibility with saved models


---

**Last Updated**: November 2024  
**Contact**: arnoldhge@gmail.com