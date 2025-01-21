#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
dataset = MultilabelDataset(X_train, y_train)
# TODO: create train loader
training_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=32)

# %% model
class MultilabelClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultilabelClassifier, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.ReLU_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()

    def forward(self, x):
        output = self.linear_1(x)
        output = self.ReLU_1(output)
        output = self.linear_2(output)
        output = self.output(output)
        return output



input_dim = X_torch.data.shape[1]
output_dim = y_torch.data.shape[1]
hidden_layer_dim = 20

model = MultilabelClassifier(input_dim=input_dim, hidden_dim=hidden_layer_dim, output_dim=output_dim)
# %% loss function, optimizer, training loop
LR = 0.01
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())
losses = []
slope, bias = [], []
number_epochs = 100

for epoch in range(number_epochs):
    for j, (X, y) in enumerate(training_loader):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

    losses.append(loss.data.detach().item())
    if (epoch%10 == 0):
        print(f"loss {epoch}th epoch: {losses[-1]}")   
    
    
    
# %% losses
sns.lineplot(x=range(number_epochs), y=losses)
# %% test the model
def normalize(x):
    for i in range(len(x)):
        x[i] = 0 if x[i] < 0.5 else 1
    return x

with torch.no_grad():
    y_hat_test = model(X_test).detach().round().numpy()
#%% Naive classifier accuracy
y_hat_test_str = [str(x) for x in y_hat_test]
count_max = Counter(y_hat_test_str).most_common()[0][1]
print(count_max)
print(f"Naive Classifier Accuracy: {count_max/len(y_hat_test_str)}")
# %% Test accuracy
acurracy_model = accuracy_score(y_test, y_hat_test)
print(acurracy_model)
# %%
