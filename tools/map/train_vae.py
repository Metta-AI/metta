import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.map.map_dataset import MapDataset


class VAE(nn.Module):
    def __init__(self, input_height=32, input_width=32, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            h1 = self.encoder[0](dummy_input)
            h_after_conv1, w_after_conv1 = h1.shape[2:]
            h2 = self.encoder[2](self.encoder[1](h1))
            h_after_conv2, w_after_conv2 = h2.shape[2:]
            flattened_size = h2.flatten().shape[0]
            unflatten_shape = (64, h_after_conv2, w_after_conv2)

        k, s, p = 4, 2, 1
        op1_h = h_after_conv1 - ((h_after_conv2 - 1) * s - 2 * p + k)
        op1_w = w_after_conv1 - ((w_after_conv2 - 1) * s - 2 * p + k)
        op2_h = input_height - ((h_after_conv1 - 1) * s - 2 * p + k)
        op2_w = input_width - ((w_after_conv1 - 1) * s - 2 * p + k)

        self.fc1 = nn.Linear(flattened_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, flattened_size),
            nn.ReLU(),
            nn.Unflatten(1, unflatten_shape),
            nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s, padding=p, output_padding=(op1_h, op1_w)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=k, stride=s, padding=p, output_padding=(op2_h, op2_w)),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h_fc1 = self.fc1(h)
        mu, logvar = self.fc2(h_fc1), self.fc3(h_fc1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a VAE on map data.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    dataset = MapDataset(args.file_path)
    if len(dataset) == 0:
        print("Dataset is empty. Check file path or file content.")
        return

    sample = dataset[0]
    height, width, _ = sample.shape
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_height=height, input_width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):
            data = data.permute(0, 3, 1, 2).to(device)
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "vae.pth")
    print("Training complete. Model saved to vae.pth")


if __name__ == "__main__":
    main()
