#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
os.getcwd()

# %% transform and load data
# set up image transforms
transformz = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
# set up train and test datasets
train_data = torchvision.datasets.ImageFolder(root="data/train", transform=transformz)
trainloader = DataLoader(dataset=train_data, shuffle=True, batch_size=4)
test_data = torchvision.datasets.ImageFolder(root="data/test", transform=transformz)
testloader = DataLoader(dataset=test_data, shuffle=True, batch_size=4)

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self, classes_num) -> None:
        super(ImageMulticlassClassificationNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 5, 3) # 48
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(5, 16, 3) # 11
        self.linear_1 = nn.Linear(16*11*11, 200)
        self.linear_2 = nn.Linear(200, 64)
        self.linear_3 = nn.Linear(64, 20)
        self.linear_4 = nn.Linear(20, classes_num)
        self.softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()


    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = self.flat(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.softmax(x)
        return x
        

input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet(classes_num=NUM_CLASSES)      
# %% loss function and optimizer
# set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# %% training
losses = []
NUM_EPOCHS = 40
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')
    losses.append(loss.item())
# %% graph
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%
