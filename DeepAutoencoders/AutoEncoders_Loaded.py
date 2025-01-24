#%% packages
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils 

#%% Dataset and data loader
path_images = 'data/train'

transform = transforms.Compose(
    [transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

dataset = ImageFolder(root=path_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# %% model class
LATENT_DIMS = 128

# TODO: Implement Encoder class
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # batch, 6, 62, 62
        self.conv2 = nn.Conv2d(6, 16, 3) # batch, 16, 60, 60
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(16*60*60, latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flat(x)
        return self.linear(x)
        



# TODO: Implement Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.t_conv_1 = nn.ConvTranspose2d(16, 6, 3)
        self.t_conv_2 = nn.ConvTranspose2d(6, 1, 3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(latent_dim, 60*60*16)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(self.t_conv_1(x.view(-1, 16, 60, 60)))
        x = self.relu(self.t_conv_2(x))
        return x
        

# TODO: Implement Autoencoder class
class AutoEncoder(nn.Module):
    def __init__(self, latent_space):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_space)
        self.decoder = Decoder(latent_dim=latent_space)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    
# Test it
input = torch.rand((1, 1, 64, 64))
model = AutoEncoder(latent_space=LATENT_DIMS)
model(input).shape


#%% init model, loss function, optimizer
model = AutoEncoder(latent_space=LATENT_DIMS)
# %% with sgd now
model.load_state_dict(torch.load("weights.pth"))

# %% visualise original and reconstructed images
def show_image(img):
    img = 0.5 * (img + 1)  # denormalizeA
    # img = img.clamp(0, 1) 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

images, labels = iter(dataloader).next()
print('original')
plt.rcParams["figure.figsize"] = (20,3)
show_image(torchvision.utils.make_grid(images))

# %% latent space
print('latent space')
latent_img = model.encoder(images)
latent_img = latent_img.view(-1, 1, 8, 16)
show_image(torchvision.utils.make_grid(latent_img))
#%%
print('reconstructed')
show_image(torchvision.utils.make_grid(model(images)))


# %% Compression rate
image_size = images.shape[2] * images.shape[3] * 1
compression_rate = (1 - LATENT_DIMS / image_size) * 100
compression_rate

torch.save(model.state_dict(), "weights.pth")

# %%
