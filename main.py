import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 512 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, 4, 4)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 2 * kl_loss


if __name__ == "__main__":

    # load dataset
    os.system("kaggle datasets download -d jessicali9530/celeba-dataset")
    os.system("unzip celeba-dataset.zip")

    # crop
    detector = MTCNN(image_size=64, device='cuda')
    for img_name in tqdm(os.listdir("img_align_celeba/img_align_celeba")):
        img = Image.open(f"img_align_celeba/img_align_celeba/{img_name}").convert("RGB")
        face = detector(img)
        if face is not None:
            face = face.permute(1, 2, 0).cpu().numpy()
            face = ((face + 1) / 2 * 255).astype('uint8')
            Image.fromarray(face).save(f"faces/train/{img_name}")
        else:
            print(f"{img_name}: face not found")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(
        root="faces",
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )
    vae = VAE().to("cuda")
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0002)
    loss_history = []
    best_loss = float('inf')
    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs = imgs.to("cuda")
            optimizer.zero_grad()
            recon, mu, logvar = vae(imgs)
            loss = vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.2f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(vae.state_dict(), "best_vae.pth")
        with torch.no_grad():
            z = torch.randn(64, 256).to("cuda")
            samples = vae.decode(z)
            save_image(samples, f"samples_{epoch + 1}.png", normalize=True)

    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.show()

    # vae = VAE().to("cuda")
    # vae.load_state_dict(torch.load("best_vae.pth"))
    # vae.eval()

    fid = FrechetInceptionDistance(feature=2048).to("cuda")
    is_score = InceptionScore().to("cuda")
    eval_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )
    num_gen = 0
    with torch.no_grad():
        for imgs, _ in eval_loader:
            imgs = imgs.to("cuda")
            real = (imgs + 1) / 2
            real = torch.nn.functional.interpolate(real, size=(299, 299))
            real = (real * 255).clamp(0, 255).to(torch.uint8)
            fid.update(real, real=True)
            z = torch.randn(imgs.size(0), 256).to("cuda")
            fake = vae.decode(z)
            fake = (fake + 1) / 2
            fake = torch.nn.functional.interpolate(fake, size=(299, 299))
            fake = (fake * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake, real=False)
            is_score.update(fake)
            num_gen += imgs.size(0)
            if num_gen >= 10000:
                break
    fid_value = fid.compute()
    is_mean, is_std = is_score.compute()
    print(f"FID: {fid_value:.4f}")
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
