#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import seaborn as sns
os.getcwd()

#%% transform, load data
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataset_train = torchvision.datasets.ImageFolder(root="data/train", transform=transform)
dataset_test = torchvision.datasets.ImageFolder(root="data/test", transform=transform)
trainloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
testloader = DataLoader(dataset=dataset_test, batch_size=4, shuffle=True) 
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images, nrow=2))
# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super(ImageClassificationNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, 3) #30
        self.max_pool = nn.MaxPool2d(2, 2) #15
        self.conv_2 = nn.Conv2d(6, 16, 3) #13 -> 6
        self.linear_1 = nn.Linear(16*6*6, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 10)
        self.linear_4 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.sigmoid(x)
        return x

#%% init model
model = ImageClassificationNet()      
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# %% training
NUM_EPOCHS = 10
losses = []
for epoch in range(NUM_EPOCHS):
    for i, (X, y) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y.reshape(-1, 1).float())
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)},'
                  f'Loss: {loss.item():.4f}')
    losses.append(loss.item())
# %% plot
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.