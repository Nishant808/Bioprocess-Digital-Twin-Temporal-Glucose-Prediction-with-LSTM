# Conv1D-LSTM Autoencoder: Example Output

## Overview
This document showcases the expected output from running `bioprocess_conv1d_lstm_autoencoder.py`. The visualization demonstrates the model's ability to reconstruct multivariate bioreactor sensor data and detect anomalies.

---

## Output Visualization Description

The script generates a 5-panel figure titled **"Conv1D-LSTM Autoencoder Performance: Continuous Biological Kinetics Tracker"** showing:

---

## Panel Descriptions

### Panel 1: Glucose Levels
- **Black solid line:** Actual biological glucose consumption dynamics
- **Purple dashed line:** Model-reconstructed glucose trajectory
- **Observation:** The model captures the exponential decay pattern with noise. During the anomaly zone (orange shaded region, t=600-750), the model shows increased divergence from actual values, indicating abnormal metabolic behavior.

### Panel 2: Dissolved Oxygen (DO %)
- **Black solid line:** Actual DO sensor readings
- **Blue dashed line:** Model reconstruction
- **Anomaly Signal:** Sharp DO collapse during t=600-750 is detected as the model struggles to reconstruct this dramatic deviation. This reflects the simulated agitation failure.

### Panel 3: pH Metrics
- **Black solid line:** Actual pH drift due to metabolite accumulation
- **Green dashed line:** Model trajectory
- **Pattern:** Gradual pH decline is well-tracked by the model under normal conditions. The anomaly region shows minor divergence.

### Panel 4: Temperature (°C)
- **Black solid line:** Actual temperature with diurnal oscillations
- **Red dashed line:** Model reconstruction
- **Anomaly Indicator:** Clear temperature spike during t=600-750 (1.5°C overshoot) is evident in both actual and reconstructed signals, though the model slightly underestimates the peak.

### Panel 5: Reconstruction Error & Anomaly Detection
- **Black solid line:** Mean Absolute Error (MAE) across all 4 variables at each timestep
- **Dark red dotted line:** System Anomaly Detection Threshold (μ + 4σ based on normal-phase error distribution)
- **Orange shaded region:** Injected malfunction zone (t=600-750)

#### Key Findings:
- **Baseline Error (t=0-600):** ~1-4 MAE units during normal operation
- **Anomaly Detection:** Clear spike in reconstruction error during t=600-750, exceeding the detection threshold
- **Sensitivity:** The threshold successfully identifies the injected anomaly without false positives in the post-anomaly recovery phase

---

## Performance Metrics

| Metric | Value |
| --- | --- |
| **Training Epochs** | 20 |
| **Batch Size** | 16 |
| **Window Size** | 24 timesteps |
| **Input Channels** | 4 (glucose, DO, pH, temperature) |
| **Encoder Channels** | 16 → 8 |
| **LSTM Hidden Dimension** | 32 |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Anomaly Detection Threshold** | μ + 4σ of reconstruction error |

---

## Interpretation & Implications

### Strengths:
1. **Multivariate Capture:** The autoencoder successfully captures trends across all 4 sensor streams simultaneously
2. **Anomaly Separation:** Clear distinction between normal operation (low error) and anomalous period (high error spike)
3. **Biological Realism:** Reconstructed trajectories maintain physical plausibility (e.g., non-negative glucose, DO bounded at 0-100%)

### Limitations:
1. **Lag in Peak Detection:** Reconstruction error peaks slightly after anomaly injection (lag ~5-10 steps), indicating temporal sensitivity
2. **Post-Anomaly Transients:** Minor error spikes appear during recovery phase (t=750-850), possibly reflecting model adaptation
3. **Synthetic Data:** Results are based on simulated bioreactor data; validation on real bioprocess sensors is needed

---

## Output Characteristics

### Actual vs. Reconstructed Trajectories:
- **Glucose:** Exponential decay from ~10 g/L → ~0.1 g/L over 1000 steps
- **DO:** Sharp drop in anomaly zone from ~60% to ~10-20%
- **pH:** Gradual drift from 7.2 → 7.0 with anomaly region showing slight deviation
- **Temperature:** Diurnal oscillations (±0.2°C) with spike to ~38.5°C during anomaly

### Error Dynamics:
- **Warm-up Phase (t=0-24):** Elevated initial error as model learns window patterns
- **Steady-State (t=24-600):** Low reconstruction error (~1-3 MAE), indicating good model fit
- **Anomaly Zone (t=600-750):** Sharp error spike to ~8-20 MAE, clear anomaly signal
- **Recovery (t=750-1000):** Gradual return to baseline with minor transients

---

## Next Steps

To improve anomaly detection:
1. **Tune Detection Threshold:** Experiment with different σ multipliers (currently 4σ) for sensitivity/specificity trade-off
2. **Multi-Scale Windows:** Test multiple window sizes to capture both fast and slow anomalies
3. **Real Data Validation:** Collect labeled anomalies from actual bioreactor systems (e.g., agitation failure, contamination, heating unit malfunction)
4. **Ensemble Approach:** Combine multiple autoencoder models trained on different anomaly types for robust detection

---

## Running the Code

To reproduce this output:

```bash
python bioprocess_conv1d_lstm_autoencoder.py
```

The script will:
1. Generate 1000 timesteps of synthetic bioreactor data
2. Inject an agitation failure anomaly (t=600-750)
3. Train the Conv1D-LSTM autoencoder for 20 epochs
4. Display the 5-panel visualization
5. Print training loss at epochs 5, 10, 15, 20

**Expected runtime:** ~2-5 minutes depending on hardware (faster on GPU/MPS)

---

## Console Output Example

```
Training the Custom Biological Autoencoder Model...
Epoch 05/20 | Train Loss: 0.015234
Epoch 10/20 | Train Loss: 0.008921
Epoch 15/20 | Train Loss: 0.006745
Epoch 20/20 | Train Loss: 0.005123
```
