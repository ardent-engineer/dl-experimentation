#%% packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import seaborn as sns
import numpy as np
#%% data import 
twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df.head()
cat_id = {'neutral': 1, 'negative': 0, 'positive': 2}
# Implement your code here
X = df['text'].values
y = df['sentiment'].map(cat_id)
y = y.values
print(y)
#%% Hyperparameters
BATCH_SIZE = 512
NUM_EPOCHS = 50
LR = 0.001
#%%
one_hot = CountVectorizer()
X = one_hot.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

#%% Dataset Class
class SentimentData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray()).float()
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
train_dataset = SentimentData(X_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

test_dataset = SentimentData(X_test, y_test)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

# %% Model
class SentimentModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN=10):
        super().__init__()
        self.linear_1 = nn.Linear(NUM_FEATURES, HIDDEN)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, NUM_CLASSES)
        self.final = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.final(x)
        return x

model = SentimentModel(26439, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# %% Model Training
train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    print(e)
    for idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)

        loss.backward()
        curr_loss += loss.item()
        optimizer.step()
    train_losses.append(curr_loss)

# %% Plot training loss
sns.lineplot(x=range(NUM_EPOCHS), y=train_losses)
# %% Model Evaluation
model.eval()
y_hat = torch.tensor(np.array([]))
for i, (X, y) in enumerate(test_loader):
    with torch.no_grad():
        y_hat = torch.cat([y_hat, model(X).argmax(dim=1)])
print(y_hat)
print(accuracy_score(y_test, y_hat))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_hat, target_names=cat_id.keys()))

# %%
