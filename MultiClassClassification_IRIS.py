#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# %% convert to float32
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
# %% dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

iris_data = IrisDataset(X_train, y_train)
# %% dataloader
training_loader = DataLoader(iris_data, batch_size=32, shuffle=True)
# %% check dims
print(f"X: {iris_data.X.shape} y: {iris_data.y.shape}")
# %% define class
class MultiClassClassifier(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x

        
# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 20
NUM_CLASSES = len(iris_data.y.unique())
# %% create model instance
model = MultiClassClassifier(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN)
# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
LR = .01
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# %% training
losses = []
NUM_EPOCHS = 1000
for epoch in range(NUM_EPOCHS):
    for X, y in training_loader:
        optimizer.zero_grad()
        y_pred_log = model(X)
        loss = criterion(y_pred_log, y)
        loss.backward()
        optimizer.step()
    losses.append(loss.data.detach().item())
print(losses)
# %% show losses over epochs
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
print(losses[-1])
# %% test the model
X_test = X_test.astype("float32")
X_test_th = torch.from_numpy(X_test)

# %% Accuracy
with torch.no_grad():
    y_pred_test = model(X_test_th)
    y_with_indices = torch.max(y_pred_test.detach(), 1)

score = accuracy_score(y_test, y_with_indices.indices)
print(score)

# %%
