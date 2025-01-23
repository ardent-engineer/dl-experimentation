#%% packages
from collections import OrderedDict 
import numpy as np 
import torch 
from torch import optim 
import torch.nn as nn 
import torchvision 
from torchvision import transforms, models 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score
from PIL import Image

# %% data prep
train_dir = 'train' 
test_dir =  'test'

transform = transforms.Compose([transforms.Resize(255), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()]) 
 
dataset = torchvision.datasets.ImageFolder(train_dir, transform= transform) 
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True) 

dataset = torchvision.datasets.ImageFolder(test_dir, transform= transform) 
test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True) 

# %% Display sample images
def imshow(image_torch): 
    image_torch = image_torch.numpy().transpose((1, 2, 0)) 
    plt.figure() 
    plt.imshow(image_torch) 
 
X_train, y_train = next(iter(train_loader)) 
image_grid = torchvision.utils.make_grid(X_train[:15, :, :, :], scale_each=True, nrow=4) 
imshow(image_grid)

#%% Model setup
# TODO: Load pre-trained DenseNet121 model
model = models.densenet121(pretrained=True)
# TODO: Freeze all model parameters
for parameter in model.features.parameters():
    parameter.requires_grad = False 
print(model)
# %%
# TODO: Replace classifier with new sequential layer containing:
#        - Linear layer with 1024 input features and 1 output feature
#        - Sigmoid activation
model.classifier = nn.Sequential(OrderedDict([
    ("out_1", nn.Linear(1024, 1)),
    ("final", nn.Sigmoid())
]))

# %% Training setup
# TODO: Define Adam optimizer for classifier parameters only
opt = torch.optim.Adam(model.classifier.parameters())
# TODO: Define appropriate loss function for binary classification
loss_function = nn.BCELoss()

train_losses = []
NUM_EPOCHS = 10

# %% Training loop
model.train()
for epoch in range(NUM_EPOCHS): 
    train_loss = 0 
    for bat, (img, label) in enumerate(train_loader): 
        # TODO: Zero the gradients
        opt.zero_grad()

        # TODO: Forward pass
        y_hat = model(img)

        # TODO: Calculate loss
        loss = loss_function(y_hat.squeeze(),label.float())

        # TODO: Backward pass
        loss.backward()

        # TODO: Update weights
        opt.step()
        train_loss += loss.item() 
        print(f"batch_{bat}/epoch_{epoch} learned")
    train_losses.append(train_loss) 
    print(f"epoch: {epoch}, train_loss: {train_loss}") 

# %% Plot losses
sns.lineplot(x=range(len(train_losses)), y=train_losses)

# %% Predictions and evaluation
fig = plt.figure(figsize=(10, 10)) 
class_labels = {0:'cat', 1:'dog'} 
X_test, y_test = next(iter(test_loader)) 

with torch.no_grad():
    # TODO: Get model predictions
    y_pred = model(X_test)
    y_pred = y_pred.round()
    y_pred = [p.item() for p in y_pred] 

for num, sample in enumerate(X_test): 
    plt.subplot(4,6,num+1) 
    plt.title(class_labels[y_pred[num]]) 
    plt.axis('off') 
    plt.imshow(np.transpose(sample.cpu().numpy(), (1,2,0))) 

# TODO: Calculate and print accuracy score
acc = accuracy_score(y_pred, y_test)
print(f"Accuracy Score: {np.round(acc * 100, 2)} %")
# %%
from PIL import Image
transform = transforms.Compose([transforms.Resize(255), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor()]) 
img_dog = Image.open("dog.jpg")
img_cat = Image.open("cat.jpg")
img_dog = transform(img_dog)
img_cat = transform(img_cat)
X_test = torch.stack([img_cat, img_dog])
y_test = torch.stack([torch.tensor(0), torch.tensor(1)])
# %%

with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.round()
    y_pred = [p.item() for p in y_pred] 

for num, sample in enumerate(X_test): 
    plt.subplot(4,6,num+1) 
    plt.title(class_labels[y_pred[num]]) 
    plt.axis('off') 
    plt.imshow(np.transpose(sample.cpu().numpy(), (1,2,0))) 
# %%
