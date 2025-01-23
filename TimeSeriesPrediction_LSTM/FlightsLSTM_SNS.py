#%% Packages
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Import
data = sns.load_dataset("flights")
print(f'Number of Entries: {len(data)}')
data.head()

# %%
sns.lineplot(data.index, data.passengers, data=data)
# %%
# Convert passenter data to float32 for PyTorch
num_points = len(data)
Xy = data.passengers.values.astype(np.float32)

#%% scale the data
scaler = MinMaxScaler()

Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))


# %% Data Restructuring
#%% train/test split
X = []
y = []
for i in range(Xy_scaled.shape[0]-10):
    X.append(Xy_scaled[i:i+10])
    y.append(Xy_scaled[i+10])
X = torch.from_numpy(np.array(X))
X = torch.from_numpy(np.array(y))
X_train, y_train = X[:-12], y[:-12]
X_test, y_test = X[-12:], y[-12:]
# %% 
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = X.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
train_set = FlightDataset(X_train, y_train)
test_set = FlightDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=16)
test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=16)
# %%
class FlightModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm_1 = nn.LSTM(input_size, hidden_size=5, num_layers=1, batch_first=True)
        self.fc_1 = nn.Linear(5, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = self.fc_1(x)
        x = self.relu(x)
        return x
# %% Model, Loss and Optimizer
model = FlightModel(1, 1)

loss_fun = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
NUM_EPOCHS = 200
losses = []
#%% Train
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fun(y_hat, y)
        loss.backward()
        total_loss += loss.detach().item()
        optimizer.step()
    losses.append(total_loss)
# %%
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# %% Create Predictions
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test_torch)
y_act = y_test_torch.numpy().squeeze()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)

# %%
