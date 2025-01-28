#%% packages
import torch
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import torchvision
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from PIL import Image
import random
from collections import OrderedDict
import seaborn as sns
# %% Hyperparameters
NUM_EPOCHS = 1000
loss_fn_self = nn.CrossEntropyLoss()
loss_fn_super = nn.BCEWithLogitsLoss()
loss_factor_self = 1
# %% image transformation steps
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),

])
labled_train_set = torchvision.datasets.ImageFolder(root="data/train", transform=transform)
labled_test_set = torchvision.datasets.ImageFolder(root="data/test", transform=transform)
labled_train_loader = DataLoader(labled_train_set, batch_size=8)
labled_test_loader = DataLoader(labled_test_set, batch_size=8)

#%% Class for Unlabeled Dataset
class UnlabeledDataset(Dataset):
    transform_map = {
        0: 0,
        1: 90,
        2: 180,
        3: 270
    }
    
    def __init__(self, folder_path, transforms):
        super().__init__()
        self.folder_files = os.listdir(folder_path)
        self.folder_files_relpath = [f"{folder_path}/{x}" for x in self.folder_files]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.folder_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.folder_files_relpath[idx])
        label = random.randint(0, 3)
        img = transforms.functional.rotate(img, angle=self.transform_map[label])
        img = self.transforms(img)
        y = [0] * 4
        y[label] = 1
        y = np.array(y)
        y = torch.from_numpy(y).float()
        return img, y


# %% Dataset and Dataloaders
unlabled_dataset = UnlabeledDataset("data/unlabeled", transforms=transform)
unlabled_dataloader = DataLoader(dataset=unlabled_dataset, batch_size=8, shuffle=True)
#%% Model Class
class SesemiNet(nn.Module):
    def __init__(self, n_super_classes, n_selfsuper_classes) -> None:
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ("conv_1", nn.Conv2d(1, 6, 3)), # 1, 30, 30
            ("relu_1", nn.ReLU()),
            ("max_1", nn.MaxPool2d(2, 2)), # 6, 15, 15
            ("conv_2", nn.Conv2d(6, 16, 3)), # 16, 13, 13  
            ("conv_3", nn.Conv2d(16, 16, 3)), # 16, 11, 11
            ("flatten", nn.Flatten()),
            ("linear_1", nn.Linear(16*11*11, 16*11)),
            ("linear_2", nn.Linear(16*11, 64))
        ]))
        self.semi_supervised_fc = nn.Linear(64, n_selfsuper_classes)
        self.supervised_fc = nn.Linear(64, n_super_classes)
    
    def forward(self, x_supervised, x_selfsupervised):
        return self.supervised_fc(self.backbone(x_supervised)), self.semi_supervised_fc(self.backbone(x_selfsupervised))

#%% Model Initialization
model = SesemiNet(len(labled_train_set.classes)-1, 4)
#%% Loss functions and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
losses = []
print(list(enumerate(labled_train_loader))[1])
# %% Training loop
train_losses_self = []
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    data_loaders = zip(labled_train_loader, unlabled_dataloader)
    
    for i, (supervised_data, selfsupervised_data) in enumerate(data_loaders):
        optimizer.zero_grad()
        X_self, y_self = selfsupervised_data
        X_super, y_super = supervised_data
        y_hat_super, y_hat_semi = model(X_super, X_self)

        loss_super =loss_fn_super(y_hat_super.squeeze(), y_super.float())
        loss_self = loss_fn_self(y_hat_semi, y_self)
        loss = loss_super + (loss_self*10000)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    losses.append(train_loss)
    print(f"{epoch}: {train_loss}")


# %% Visualization
sns.lineplot(x=range(len(losses)), y=losses)
# %% Testing
y_test_preds = []
y_test_trues = []
with torch.no_grad():
    for (X_test, y_test) in labled_test_loader:
         y_test_pred = model(X_test, X_test) 
         y_test_pred_argmax = torch.argmax(y_test_pred[0], axis = 1)
         y_test_preds.extend(y_test_pred_argmax.numpy())
         y_test_trues.extend(y_test.numpy())
# %%
accuracy_score(y_pred=y_test_preds, y_true=y_test_trues)
    
# %%
