# Bioprocess Digital Twin: Temporal Glucose Prediction with LSTM

## 🧬 Project Overview

This repository contains a deep learning pipeline designed to predict **Glucose concentration** in a bioreactor based on real-time online sensor data (DO, pH, and Temperature). The project explores multiple neural network architectures, from baseline LSTM models to advanced hybrid approaches combining **Convolutional Neural Networks (Conv1D) with LSTM autoencoders** for anomaly detection and process monitoring.

## 🛠 Tech Stack

* **Language:** Python 3.11+
* **Deep Learning Framework:** PyTorch (with Metal Performance Shaders - MPS support)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib

## 🏗 Model Architectures

### Phase I: Temporal LSTM Regression
* **Type:** Recurrent Neural Network (LSTM)
* **Input Shape:** `(Batch, 10, 3)` — 10-hour lookback window with 3 online features.
* **Hidden Layers:** 64 units (Standard configuration)
* **Output:** Linear layer with optional constraints (ReLU/Sigmoid) for concentration regression.
* **Optimization:** Adam Optimizer with Learning Rate Scheduling (`ReduceLROnPlateau`).

### Phase II: Conv1D-LSTM Autoencoder (NEW)
* **Type:** Hybrid Convolutional-Recurrent Autoencoder
* **Encoder:** 
  - Conv1D layers (3 → 16 → 8 channels) for spatial feature extraction
  - LSTM layer (8 → 32 hidden units) for temporal compression
* **Decoder:**
  - LSTM layer (32 → 8 hidden units) for temporal reconstruction
  - Dense layer for feature reconstruction
* **Task:** Unsupervised learning for anomaly detection and multivariate process monitoring
* **Input Shape:** `(Batch, Window, 4 Features)` — Multi-step sliding windows capturing glucose, DO, pH, and temperature dynamics
* **Window Size:** 24 timesteps (configurable)
* **Loss Function:** Mean Squared Error (MSE) for reconstruction quality

---

## 🔬 Experimental Journey & Iteration Log

The following table documents the systematic optimization process and the challenges encountered during model convergence.

| Iteration | Configuration | Key Metrics | Observation & Result |
| --- | --- | --- | --- |
| **01** | Baseline: 1k Epochs, LR=1e-4 | MAE: 0.10, $R^2$: -0.46 | **Mode Collapse.** Model predicted the global mean. |
| **02** | Output Constraint (ReLU/Sigmoid) | MAE: 0.17, $R^2$: -3.30 | Prevented negatives but stayed in a local minimum. |
| **03** | High LR (0.003) + Scheduler | MAE: 1.28, $R^2$: -186.1 | **Gradient Divergence.** High bias introduced. |
| **04** | Target Clipping & Inverse Dummy | MAE: 0.49, $R^2$: -22.3 | Solved scaling/negative glucose artifact. |
| **05** | Increased Hidden Size (128) | Variable | Investigated model capacity vs. data signal. |
| **06** | Conv1D-LSTM Autoencoder | Under Testing | Novel architecture combining spatial & temporal modeling for anomaly detection. |

### 🔍 Key Technical Insights

* **Hardware Acceleration:** Successfully implemented the **MPS (Metal Performance Shaders)** backend, reducing training time significantly on M4 architecture compared to CPU-bound training.
* **The "Mean Guessing" Problem:** Identified a persistent mode collapse where the model converged to the mean of the training set. This highlighted the need for higher signal-to-noise ratios in supervised learning.
* **Inverse Scaling Nuances:** Developed a robust "Dummy Array" method to perform inverse transformations on 1D predictions using a 4D scaler, ensuring real-world unit integrity.
* **Hybrid Architecture Benefits:** The Conv1D-LSTM autoencoder offers:
  - **Spatial Feature Extraction:** Conv1D layers capture local patterns in multivariate sensor data
  - **Temporal Context:** LSTM layers model long-range dependencies in cultivation dynamics
  - **Unsupervised Anomaly Detection:** Reconstruction error serves as an anomaly score without labeled anomalies
  - **Interpretability:** Clear separation of normal vs. anomalous operation phases

---

## 📈 Current Status & Roadmap

While the original LSTM regression $R^2$ remains negative, the project has successfully mapped out the **failure surface** of temporal regression in bioprocesses. The low MAE (0.10 g/L) indicates the model captures meaningful patterns but struggles with absolute prediction.

### Phase II: Conv1D-LSTM Autoencoder
The new autoencoder approach shifts focus from supervised glucose prediction to **unsupervised process monitoring**:
- Detects anomalies (e.g., agitation failures, temperature deviations) via reconstruction error
- Captures multivariate relationships between glucose, DO, pH, and temperature
- Operates on synthetic bioreactor data with injected anomalies (t=600-750)
- Empirical anomaly threshold: μ + 4σ of normal-phase reconstruction error

### Next Steps for Phase III:

1. **Labeled Anomaly Dataset:** Collect or generate diverse failure modes with ground truth labels
2. **Supervised Anomaly Classification:** Add a classification head to the autoencoder
3. **Log-Space Transformation:** Moving to $log(y+1)$ to amplify gradients at low concentrations
4. **Delta-Prediction:** Reframing the task to predict $\Delta$ Glucose rather than absolute concentration
5. **Feature Engineering:** Incorporating rolling means or rates of change as secondary features
6. **Transfer Learning:** Pre-train on synthetic data; fine-tune on real bioreactor datasets

---

## 📂 Repository Structure

```
.
├── README.md                                    (This file)
├── Bioprocess_LSTM.ipynb                       (Phase I: Baseline LSTM notebook)
├── bioprocess_conv1d_lstm_autoencoder.py       (Phase II: New autoencoder implementation)
├── requirements.txt                            (Python dependencies)
└── [Other project files...]
```

---

## 🚀 How to Run

### Phase I: Baseline LSTM
1. Ensure you have a PyTorch-compatible M-series Mac (or adjust device settings)
2. Clone the repo and install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Bioprocess_LSTM.ipynb`

### Phase II: Conv1D-LSTM Autoencoder (NEW)
1. Install dependencies (if not already done): `pip install -r requirements.txt`
2. Run the script: `python bioprocess_conv1d_lstm_autoencoder.py`
3. The script will:
   - Generate synthetic bioreactor data with injected anomalies
   - Train the Conv1D-LSTM autoencoder for 20 epochs
   - Reconstruct sensor timelines and compute anomaly scores
   - Visualize predictions vs. actual values with anomaly detection threshold

### Expected Output
- 5-panel matplotlib figure showing:
  - Glucose levels (actual vs. model trajectory)
  - Dissolved Oxygen dynamics
  - pH trends
  - Temperature monitoring
  - Reconstruction error with anomaly detection threshold
![Image](/output.png)
---

## 🔧 Customization

Edit parameters in `bioprocess_conv1d_lstm_autoencoder.py`:
```python
WINDOW_SIZE = 24          # Adjust sliding window length
NUM_FEATURES = 4          # Number of sensor channels
EPOCHS = 20               # Training iterations
BATCH_SIZE = 16           # Batch size for DataLoader
hidden_dim = 32           # LSTM hidden dimension
```

---

## 📚 References & Methodology

This project implements concepts from:
- **LSTM Time Series Forecasting:** Hochreiter & Schmidhuber (1997)
- **Autoencoder-based Anomaly Detection:** Chalapathy et al. (2018)
- **Hybrid CNN-LSTM Architectures:** Karim et al. (2017)
- **Bioprocess Monitoring:** Real-time sensor data fusion in fed-batch fermentation

---

## 📞 Contact & Contributions

For questions, suggestions, or collaboration opportunities, please open an issue or reach out via GitHub.

**Project Status:** Active Development (Phase II in progress)
**Last Updated:** June 2026
