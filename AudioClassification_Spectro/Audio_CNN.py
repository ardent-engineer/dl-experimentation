#%% package import
import torchaudio
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns   
import matplotlib.pyplot as plt
import os
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix

torchaudio.info
# %% preprocess and load
preprocess = transforms.Compose(
    [transforms.Resize((100,100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

dataset = torchvision.datasets.ImageFolder(root="data", transform=preprocess)
trainset_size = int(len(dataset)*.8)
testset_size = len(dataset) - trainset_size
dataset_train, dataset_test = random_split(dataset=dataset, lengths=[trainset_size, testset_size])
# %% data import
train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=4, shuffle=True)
CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']
# %% modeling
class AudioClassifierNet(nn.Module):
    def __init__(self, no_classes):
        super(AudioClassifierNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 6, 3) # 98 
        self.max_pool = nn.MaxPool2d(2, 2) # 49
        self.conv_2 = nn.Conv2d(6, 16, 3) # 23
        self.relu = nn.ReLU()
        self.flat = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(16*23*23, 256)
        self.linear_2 = nn.Linear(256, 10)
        self.linear_3 = nn.Linear(10, no_classes)
        self.output = nn.LogSoftmax()

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
        return x
# %%
model = AudioClassifierNet(len(CLASSES))
# %% hyper params

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 50

# %% training loop
losses = []
for epoch in range(NUM_EPOCHS):
    for idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.detach().item())
    print(f"epoch {epoch}th loss: {losses[-1]}")

sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# %%
y_test = []
y_test_hat = []
for i, data in enumerate(test_loader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        outputs = model(inputs)
        y_test_hat_temp = torch.argmax(outputs, dim=1)  # Corrected: use argmax
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_hat)  # Directly compare without argmax
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_hat)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# %%
