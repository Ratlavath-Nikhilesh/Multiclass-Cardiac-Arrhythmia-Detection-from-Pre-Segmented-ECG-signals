# Multiclass Cardiac Arrhythmia Detection from Pre-Segmented ECG Signals  

## Overview
This project focuses on the automated classification of heartbeats from pre-segmented ECG signals using both classical machine learning and deep learning approaches.  
Accurate heartbeat classification is a critical step in diagnosing cardiac arrhythmias, but manual ECG interpretation is time-consuming and error-prone. This work aims to build robust models that can reliably classify multiple heartbeat types directly from ECG signals while handling challenges such as class imbalance and signal variability.

This project was carried out as part of **Research Internship** at **Indian Institute of Technology Bhubaneswar**, under the supervision of **Dr. Barathram Ramkumar**.

---

## Problem Statement
Cardiac arrhythmia diagnosis traditionally relies on manual ECG interpretation by clinicians. This process is slow, prone to human error, and becomes increasingly unreliable when dealing with large volumes of data.  
Additionally, ECG datasets often suffer from **class imbalance** and **signal noise**, which limits the effectiveness of conventional machine learning models.

The objective of this project is to develop **automated and robust models** capable of accurately classifying multiple arrhythmic heartbeat types from ECG signals.

---

## Dataset
- **Source**: MIT-BIH Arrhythmia Database (https://www.physionet.org/content/mitdb/1.0.0/)
- **Total Samples**:
  - Training set: 87,554 heartbeat samples  
  - Test set: 21,892 heartbeat samples
- **Signal Length**: 187 timesteps per heartbeat
- **Number of Classes**: 5

### Heartbeat Classes
- **N** – Normal
- **S** – Supraventricular ectopic beat
- **V** – Ventricular ectopic beat
- **F** – Fusion beat
- **Q** – Unknown / other beats

Each sample represents a single pre-segmented heartbeat extracted from long ECG recordings.

---

## Data Preprocessing
The following preprocessing steps were applied before model training:

- **Signal normalization** using per-beat Z-score normalization
- **Stratified train–validation split** to preserve class distribution
- **Class imbalance handling**
  - Inverse-frequency class weights for deep learning models
  - Automatic class-weight balancing for Scikit-learn models

These steps ensured stable training and fair learning across all heartbeat categories.

---

## Models Implemented

### Classical Machine Learning Models
- **Logistic Regression**
  - Multiclass classification using balanced class weights
- **Random Forest**
  - Hyperparameter tuning performed using Optuna
- **XGBoost**
  - Optuna-based hyperparameter tuning with early stopping and class-weighted training

All classical models were evaluated using accuracy, per-class metrics, and macro-level performance measures.

---

### Deep Learning Models

#### 1D Convolutional Neural Network (1D CNN)
- Wide-kernel Conv1D layers for full-beat morphology capture
- Multiple depthwise-separable residual blocks
- Squeeze-and-Excitation (SE) attention modules
- Progressive channel expansion and temporal downsampling
- Global Average Pooling followed by softmax classification

#### Hybrid CNN–BiLSTM Model
- Convolutional stem with residual SE blocks and dilated convolutions
- Temporal modeling using stacked Bidirectional LSTM layers
- Residual connections to preserve temporal features
- Multi-statistic temporal pooling (Average, Max, Standard Deviation)
- Dropout regularization and softmax output

This hybrid architecture combines spatial feature extraction with temporal dependency modeling.

---

## Model Evaluation
All models were evaluated on a validation set using consistent metrics.  
Among all approaches, the **Hybrid CNN–BiLSTM model** achieved the best overall performance.

- **Final Selected Model**: Hybrid CNN–BiLSTM  
- **Key Result**: Achieved a **macro recall of approximately 0.93**, indicating strong balanced performance across all heartbeat classes, including minority classes.

---

## Tools & Technologies
- **Programming Language**: Python  
- **Libraries & Frameworks**:
  - NumPy, Pandas
  - Scikit-learn
  - TensorFlow / Keras
  - XGBoost
  - Optuna
- **Platform**: Jupyter Notebook
- 
---

## Conclusion
This project demonstrates that deep learning models, particularly hybrid architectures combining CNNs and BiLSTMs, significantly outperform classical machine learning approaches for ECG heartbeat classification.  
Effective preprocessing, class imbalance handling, and architectural design played a crucial role in achieving high and stable performance across all heartbeat categories.

The results highlight the potential of automated ECG analysis systems to support clinical decision-making and reduce diagnostic workload.

---

## Acknowledgements
I would like to express my sincere gratitude to **Dr. Barath Ramkumar** for his guidance and support throughout this internship, and to **IIT Bhubaneswar** for providing the research environment and resources necessary for this work.
