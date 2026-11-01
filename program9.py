"""Least Squares GAN (LSGAN) demo trained briefly on MNIST."""
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

Z_DIM, BATCH_SIZE = 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transform),
                    batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(),
                                 nn.Linear(128, img_dim), nn.Tanh())
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(img_dim, 128), nn.LeakyReLU(0.2),
                                 nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x)

G, D = Generator().to(device), Discriminator().to(device)
criterion = nn.MSELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(1):
    for step, (real, _) in enumerate(loader):
        real = real.view(-1, 28*28).to(device)
        batch = real.size(0)
        z = torch.randn(batch, Z_DIM, device=device)
        fake = G(z).detach()
        d_real, d_fake = D(real), D(fake)
        loss_D = 0.5 * (criterion(d_real, torch.ones_like(d_real)) +
                        criterion(d_fake, torch.zeros_like(d_fake)))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        z = torch.randn(batch, Z_DIM, device=device)
        fake = G(z)
        d_fake = D(fake)
        loss_G = 0.5 * criterion(d_fake, torch.ones_like(d_fake))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if step % 200 == 0:
            print(f"Step {step}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")
            with torch.no_grad():
                samples = G(torch.randn(16, Z_DIM, device=device)).cpu().view(-1, 1, 28, 28)
                grid = make_grid(samples, nrow=4, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
                plt.axis("off"); plt.show()
            break
    break
