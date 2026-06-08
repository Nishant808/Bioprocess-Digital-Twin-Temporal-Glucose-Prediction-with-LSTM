import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# STEP 1: GENERATE YOUR BIOLOGICAL DATASET
# ==========================================
def generate_custom_bioreactor_data():
    t = np.arange(1000)

    # 1. Biological Logic: Glucose starts at 10 and is consumed over time
    glucose = 10 * np.exp(-0.005 * t) + np.random.normal(0, 0.1, 1000)

    # 2. DO (Dissolved Oxygen): Drops as the "cells" grow
    do = 100 * np.exp(-0.003 * t) + np.random.normal(0, 0.5, 1000)

    # 3. pH: Slowly drifts down as metabolites accumulate
    ph = 7.2 - (0.0002 * t) + np.random.normal(0, 0.02, 1000)

    # 4. Temp: Stays around 37 but has a "diurnal" oscillation
    temp = 37 + 0.2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.05, 1000)
    
    raw_matrix = np.stack([glucose, do, ph, temp], axis=1)
    
    # Inject an anomaly from t=600 to t=750 
    # (e.g., Agitation failure: DO plunges sharply to near zero, Temp overheats due to metabolic stagnation)
    anomaly_start, anomaly_end = 600, 750
    raw_matrix[anomaly_start:anomaly_end, 1] -= 30.0 + np.random.normal(0, 2.0, anomaly_end - anomaly_start) # DO collapse
    raw_matrix[anomaly_start:anomaly_end, 1] = np.clip(raw_matrix[anomaly_start:anomaly_end, 1], 0, 100) # Clamp DO at zero minimum
    raw_matrix[anomaly_start:anomaly_end, 3] += 1.5 + np.random.normal(0, 0.2, anomaly_end - anomaly_start) # Temperature spike
    
    return raw_matrix, (anomaly_start, anomaly_end)

# ==========================================
# STEP 2: SLIDING WINDOW DATASET
# ==========================================
class BioreactorDataset(Dataset):
    def __init__(self, data, window_size=24):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size, :]
        return window, window

# ==========================================
# STEP 3: CONV1D-LSTM AUTOENCODER
# ==========================================
class BioAutoencoder(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim=32):
        super(BioAutoencoder, self).__init__()
        self.window_size = window_size
        
        # Conv1D expects input channels = features
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.encoder_lstm = nn.LSTM(input_size=8, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=8, num_layers=1, batch_first=True)
        self.reconstruct = nn.Linear(8, num_features)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        
        # Spatial processing via CNN
        x_cnn = x.transpose(1, 2)
        features = self.encoder_cnn(x_cnn)
        features = features.transpose(1, 2)
        
        # Temporal compression via LSTM
        _, (hn, _) = self.encoder_lstm(features)
        latent = hn[-1]
        
        # Reconstruct timeline
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        
        return self.reconstruct(decoder_out)

# ==========================================
# STEP 4: PREPARATION AND MODEL TRAINING
# ==========================================
# Parameters
WINDOW_SIZE = 24  # 24-step sliding memory context
NUM_FEATURES = 4
EPOCHS = 20
BATCH_SIZE = 16

raw_data, anomaly_bounds = generate_custom_bioreactor_data()

# MinMax Scaling to keep trend values safely between 0 and 1 for neural networks
data_min = raw_data[:500, :].min(axis=0) # Base scaling parameters entirely on normal initial cultivation period
data_max = raw_data[:500, :].max(axis=0)
scaled_data = (raw_data - data_min) / (data_max - data_min + 1e-5)

# Initialize Pytorch structures
dataset = BioreactorDataset(scaled_data, window_size=WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = BioAutoencoder(window_size=WINDOW_SIZE, num_features=NUM_FEATURES, hidden_dim=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training execution loop
model.train()
print("Training the Custom Biological Autoencoder Model...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {total_loss/len(dataloader):.6f}")

# ==========================================
# STEP 5: PREDICTION MAPPING & VISUALIZATION
# ==========================================
model.eval()
predictions_list = []

with torch.no_grad():
    for batch_x, _ in dataloader:
        preds = model(batch_x)
        predictions_list.append(preds.numpy())

# Flatten sliding windows back to a simple continuous 1000-step array
reconstructed_scaled = np.zeros_like(scaled_data)
idx = WINDOW_SIZE
for batch_pred in predictions_list:
    for window in batch_pred:
        if idx < len(scaled_data):
            reconstructed_scaled[idx, :] = window[-1, :]
            idx += 1

# Reverse MinMax transform to get original engineering values
unscaled_predictions = (reconstructed_scaled * (data_max - data_min + 1e-5)) + data_min

# Calculate point-by-point Mean Absolute Error across the 4 variables
reconstruction_error = np.mean(np.abs(raw_data - unscaled_predictions), axis=1)

# Plotting
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
titles = ['Glucose Levels', 'Dissolved Oxygen (DO %)', 'pH Metrics', 'Temperature (°C)']
colors = ['purple', 'blue', 'green', 'red']

for i in range(NUM_FEATURES):
    axes[i].plot(raw_data[:, i], label='Actual Biological Flow', color='black', alpha=0.4, lw=1.5)
    axes[i].plot(unscaled_predictions[:, i], label='Model Dynamic Trajectory', color=colors[i], linestyle='--', lw=1.5)
    axes[i].axvspan(anomaly_bounds[0], anomaly_bounds[1], color='orange', alpha=0.15, label='Injected Malfunction Zone')
    axes[i].set_ylabel(titles[i])
    axes[i].grid(True, alpha=0.2)
    if i == 0:
        axes[i].legend(loc='upper right')

# Plot Anomaly Scoring
axes[4].plot(reconstruction_error, color='black', label='Reconstruction Error (MAE Metrics)')
axes[4].axvspan(anomaly_bounds[0], anomaly_bounds[1], color='orange', alpha=0.15)
# Set empirical threshold based on initial normal operations phase
static_threshold = np.mean(reconstruction_error[:500]) + (4 * np.std(reconstruction_error[:500]))
axes[4].axhline(y=static_threshold, color='darkred', linestyle=':', label='System Anomaly Detection Threshold')
axes[4].set_ylabel('Anomaly Metric Score')
axes[4].set_xlabel('Timeline Steps (Cultivation Progression)')
axes[4].legend(loc='upper right')
axes[4].grid(True, alpha=0.2)

plt.suptitle('Conv1D-LSTM Autoencoder Performance: Continuous Biological Kinetics Tracker', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
