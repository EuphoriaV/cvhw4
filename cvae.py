import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm


class FaceDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        df = pd.read_csv(csv_file)
        self.df = df[df["image_id"].apply(lambda x: os.path.exists(os.path.join(root, x)))]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        male = 1 if row["Male"] == 1 else 0
        male = torch.tensor([male], dtype=torch.float32)
        img = self.transform(img)
        return img, male


class VAE(nn.Module):
    def __init__(self, latent_dim=256, cond_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

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
        self.fc_mu = nn.Linear(512 * 4 * 4 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4 + cond_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, 512 * 4 * 4)
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

    def encode(self, x, c):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, c], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        z = torch.cat([z, c], dim=1)
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, 4, 4)
        x_recon = self.dec(h)
        return x_recon

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 2 * kl_loss


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FaceDataset(
        root="faces/train",
        csv_file="list_attr_celeba.csv",
        transform=transform
    )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=128,
    #     shuffle=True,
    # )
    # vae = VAE().to("cuda")
    # optimizer = torch.optim.Adam(vae.parameters(), lr=0.0002)
    # loss_history = []
    # best_loss = float('inf')
    # epochs = 30
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     for imgs, male in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
    #         imgs = imgs.to("cuda")
    #         male = male.to("cuda")
    #         optimizer.zero_grad()
    #         recon, mu, logvar = vae(imgs, male)
    #         loss = vae_loss(recon, imgs, mu, logvar)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     epoch_loss /= len(dataset)
    #     loss_history.append(epoch_loss)
    #     print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.2f}")
    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         torch.save(vae.state_dict(), "best_cvae.pth")
    #     with torch.no_grad():
    #         z = torch.randn(64, 256).to("cuda")
    #         c = torch.ones(64, 1).to("cuda")
    #         samples = vae.decode(z, c)
    #         save_image(samples, f"samples_{epoch + 1}.png", normalize=True)
    #
    # plt.plot(loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("VAE Training Loss")
    # plt.show()

    vae = VAE().to("cuda")
    vae.load_state_dict(torch.load("best_cvae.pth"))
    vae.eval()
    #
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
            cond = torch.randint(0, 2, (imgs.size(0), 1)).float().to("cuda")
            fake = vae.decode(z, cond)
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
