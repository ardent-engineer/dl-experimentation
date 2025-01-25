#%% packages
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

#%% Data Preparation
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])
img = Image.open("heart.png")
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

#%% Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (1, 75, 75)
            nn.ConvTranspose2d(1, 64, 4, 2, 1, bias=False),  # Output: (64, 150, 150)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output: (32, 300, 300)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),  # Output: (1, 600, 600)
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((75, 75))  # Ensure output matches desired size
        )

    def forward(self, x):
        return self.main(x)

#%% Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (1, 75, 75)
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # Output: (64, 37, 37)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (128, 18, 18)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (256, 9, 9)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

#%% Initialize Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
dis = Discriminator().to(device)

# Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

#%% Training Loop
num_epochs = 500
batch_size = 32
losses_gen = []
losses_dis = []

for epoch in range(num_epochs):
    # Real data
    real_imgs = img_tensor.repeat(batch_size, 1, 1, 1).to(device)
    real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing
    
    # Fake data
    noise = torch.randn(batch_size, 1, 75, 75, device=device)
    fake_imgs = gen(noise)
    fake_labels = torch.zeros((batch_size, 1), device=device)
    
    # Train Discriminator
    dis_opt.zero_grad()
    
    # Real loss
    output_real = dis(real_imgs)
    loss_real = criterion(output_real, real_labels)
    
    # Fake loss
    output_fake = dis(fake_imgs.detach())
    loss_fake = criterion(output_fake, fake_labels)
    
    # Total loss
    loss_dis = (loss_real + loss_fake) / 2
    loss_dis.backward()
    dis_opt.step()
    
    # Train Generator
    gen_opt.zero_grad()
    
    # Generator loss
    output_gen = dis(fake_imgs)
    loss_gen = criterion(output_gen, real_labels)  # Trick generator
    loss_gen.backward()
    gen_opt.step()
    
    # Record losses
    losses_gen.append(loss_gen.item())
    losses_dis.append(loss_dis.item())
    
    # Progress monitoring
    if epoch % 2 == 0:
        print(f'Epoch {epoch}/{num_epochs} | D Loss: {loss_dis.item():.4f} | G Loss: {loss_gen.item():.4f}')
        with torch.no_grad():
            sample = gen(torch.randn(1, 1, 75, 75, device=device)).cpu()
            plt.imshow(sample.squeeze().numpy(), cmap='gray')
            plt.show()

#%% Final Output
with torch.no_grad():
    generated = gen(torch.randn(1, 1, 75, 75, device=device)).cpu()
    plt.imshow(generated.squeeze().numpy(), cmap='gray')
    plt.title('Generated Heart')
    plt.axis('off')
    plt.show()

# Plot training progress
plt.figure(figsize=(10,5))
plt.title("Training Progress")
plt.plot(losses_gen, label="Generator Loss")
plt.plot(losses_dis, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()