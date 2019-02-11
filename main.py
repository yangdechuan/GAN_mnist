import os
import argparse
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None, help="Unique experimental name.")
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
args = parser.parse_args()

LATENT_SIZE = 64
HIDDEN_SIZE = 256
IMAGE_SIZE = 784


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def main():
    exam_name = args.name if args.name else "bs{}-lr{}".format(args.batch_size, args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sample dir
    sample_dir = "sample-{}".format(exam_name)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # tensorboard log dir
    log_dir = os.path.join("log", exam_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # image processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),  # 3 for rgb channels
                             std=(0.5,))
    ])

    # mnist dataset and data loader
    mnist = torchvision.datasets.MNIST(root="data",
                                       train=True,
                                       transform=transform,
                                       download=True)
    data_loader = DataLoader(mnist,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)

    # Discriminator
    D = nn.Sequential(
        nn.Linear(IMAGE_SIZE, HIDDEN_SIZE),
        nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN_SIZE, 1),
        nn.Sigmoid(),
    )

    # Generator
    G = nn.Sequential(
        nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, IMAGE_SIZE),
        nn.Tanh(),
    )

    # Device setting
    D.to(device)
    G.to(device)

    # Binary cross entropy loss and optimizer.
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr)

    # Training
    total_step = len(data_loader)
    for epoch in range(args.epochs):
        for i, (images, _) in enumerate(data_loader):
            tmp_batch_size = images.size(0)
            # images shape (B, 1, 28, 28)
            # reshaped images shape (B, 784)
            images = images.reshape(tmp_batch_size, -1).to(device)  # (B, 784)
            real_labels = torch.ones(tmp_batch_size, 1).to(device)  # (B, 1)
            fake_labels = torch.zeros(tmp_batch_size, 1).to(device)  # (B, 1)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images.
            out = D(images)  # (B, 1)
            d_loss_real = criterion(out, real_labels)
            real_score = out

            # Compute BCE_Loss using fake images.
            z = torch.randn(tmp_batch_size, LATENT_SIZE).to(device)  # (B, 64)
            fake_images = G(z)  # (B, 784)
            out = D(fake_images)
            d_loss_fake = criterion(out, fake_labels)
            fake_score = out

            # Back propagation and optimize.
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(tmp_batch_size, LATENT_SIZE).to(device)
            fake_images = G(z)
            out = D(fake_images)
            g_loss = criterion(out, real_labels)

            # Back propagation and optimize.
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Add loss to summary.
            writer.add_scalar("d_loss", scalar_value=d_loss.item(), global_step=i + epoch * total_step)
            writer.add_scalar("g_loss", scalar_value=g_loss.item(), global_step=i + epoch * total_step)

            if (i + 1) % 200 == 0:
                print("epoch [{}/{}], step [{}/{}] d_loss: {:.4f}, g_loss: {:.2f}, D(x): {:.2f}, D(G(X)): {:.2f}".format(
                    epoch, args.epochs, i+1, total_step,
                    d_loss.item(), g_loss.item(),
                    real_score.mean().item(), fake_score.mean().item(),
                ))

        # Save real images.
        if epoch == 0:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, "real_images.png"))

        # Save sampled images.
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, "fake_images-{}.png".format(epoch + 1)))

    torch.save(G.state_dict(), "G.ckpt")
    torch.save(D.state_dict(), "D.ckpt")


if __name__ == "__main__":
    main()
