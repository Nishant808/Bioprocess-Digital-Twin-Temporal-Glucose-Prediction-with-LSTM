# Bioprocess Digital Twin: Temporal Glucose Prediction with LSTM

## 🧬 Project Overview

This repository contains a deep learning pipeline designed to predict **Glucose concentration** in a bioreactor based on real-time online sensor data (DO, pH, and Temperature). The project explores the challenges of using Long Short-Term Memory (LSTM) networks to capture biological kinetics and the technical nuances of deploying ML models on **Apple Silicon (M4/MPS)** hardware.

## 🛠 Tech Stack

* **Language:** Python 3.11+
* **Deep Learning Framework:** PyTorch (with Metal Performance Shaders - MPS support)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib

## 🏗 Model Architecture

* **Type:** Recurrent Neural Network (LSTM)
* **Input Shape:** `(Batch, 10, 3)` — 10-hour lookback window with 3 online features.
* **Hidden Layers:** 64 units (Standard configuration)
* **Output:** Linear layer with optional constraints (ReLU/Sigmoid) for concentration regression.
* **Optimization:** Adam Optimizer with Learning Rate Scheduling (`ReduceLROnPlateau`).

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

### 🔍 Key Technical Insights

* **Hardware Acceleration:** Successfully implemented the **MPS (Metal Performance Shaders)** backend, reducing training time significantly on M4 architecture compared to CPU-bound training.
* **The "Mean Guessing" Problem:** Identified a persistent mode collapse where the model converged to the mean of the training set. This highlighted the need for higher signal-to-noise ratios in synthetic data generation.
* **Inverse Scaling nuances:** Developed a robust "Dummy Array" method to perform inverse transformations on 1D predictions using a 4D scaler, ensuring real-world unit integrity.

---

## 📈 Current Status & "Scars"

While the current $R^2$ remains negative, the project has successfully mapped out the **failure surface** of temporal regression in bioprocesses. The low MAE (0.10 g/L) indicates the model is mathematically "close," but the negative $R^2$ proves it has yet to capture the directional variance of the glucose decay curve.

### Next Steps for Phase II:

1. **Log-Space Transformation:** Moving to $log(y+1)$ to amplify gradients at low concentrations.
2. **Delta-Prediction:** Reframing the task to predict $\Delta$ Glucose rather than absolute concentration.
3. **Feature Engineering:** Incorporating rolling means or rates of change as secondary features.

---

## 🚀 How to Run

1. Ensure you have a PyTorch-compatible M-series Mac.
2. Clone the repo and install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Bioprocess_LSTM.ipynb`

---
