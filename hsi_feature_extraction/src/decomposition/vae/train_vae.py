import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)
import numpy as np
from os import path
import matplotlib.pyplot as plt
from demo.data import retrieve_square_images
from src.decomposition.vae.simple_vae import SimpleVAE
from src.utils.consts import LIB_PARENT_PATH

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def get_dataset(
    n: int, data_path: str, label_path: str
) -> tuple[TensorDataset]:
    data = np.load(data_path, mmap_mode="r")
    label = np.load(label_path, mmap_mode="r")
    data, label = retrieve_square_images(n, data, label)
    data = torch.tensor(data)
    label = torch.tensor(label)
    return TensorDataset(data, label)


def train(
    model: SimpleVAE,
    optimizer,
    train_dataset,
    val_dataset,
    epochs=50,
    transform=None,
    device="cuda",
):
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)
    best_val_loss = 10**15
    loss_for_plot = []
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for x, label in train_dl:
            label = label.to(device)
            x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            x = transform(x)
            results = model(x)
            loss, _, _ = model.loss_function(
                x, results, M_N=x.size(0) / len(train_dataset)
            )  # 元論文ではM_N=batch_size/self.len_train_ds,
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu()

        train_loss /= len(train_dl)
        print(f"epoch: {epoch}, train loss: {train_loss}")

        val_loss = 0
        with torch.no_grad():
            for x, label in val_dl:
                label = label.to(device)
                x = x.permute(0, 3, 1, 2).to(
                    device=device, dtype=torch.float32
                )
                x = transform(x)
                results = model(x)
                loss, _, _ = model.loss_function(
                    x, results, M_N=x.size(0) / len(val_dataset)
                )
                val_loss += loss.detach().cpu()

            val_loss /= len(val_dl)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_param.pth")
        loss_for_plot.append([epoch + 1, train_loss, val_loss])
        print(f"epoch: {epoch}, valid loss: {val_loss}")
    torch.save(model.state_dict(), "last_param.pth")
    plot_loss(loss_for_plot)


def plot_loss(loss_for_plot):
    loss_for_plot = np.array(loss_for_plot)
    plt.plot(loss_for_plot[:, 0], loss_for_plot[:, 1])
    plt.plot(loss_for_plot[:, 0], loss_for_plot[:, 2])
    plt.legend(["train", "valid"])
    plt.xlabel("epoch")
    plt.ylabel("reconstruct loss + kallback leibler loss")
    plt.show()
    plt.savefig("vae_training_loss.png")


def evaluate(model, test_dataset, transform, device):
    model.load_state_dict(torch.load("best_param.pth"))
    model.eval()
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        test_loss = 0
        for x, label in test_dl:
            label = label.to(device)
            x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            x = transform(x)
            results = model(x)
            loss, _, _ = model.loss_function(
                x, results, M_N=x.size(0) / len(test_dataset)
            )
            test_loss += loss.detach().cpu()
        test_loss /= len(test_dl)
        print(f"test loss: {test_loss}")


def normalize_images(x):
    """
    x: (B, C, H, W)
    """
    return (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(
        dim=(2, 3), keepdim=True
    )


def main():
    n = 32
    data_path = path.join(LIB_PARENT_PATH, "demo", "indianpinearray.npy")
    label_path = path.join(LIB_PARENT_PATH, "demo", "IPgt.npy")
    all_dataset = get_dataset(n, data_path, label_path)
    # データセットを分割
    size = len(all_dataset)
    train_size = int(0.8 * len(all_dataset))
    val_size = int(0.1 * len(all_dataset))
    test_size = size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        all_dataset, [train_size, val_size, test_size]
    )
    device = "cuda"
    image_size = (n, n)
    vae = SimpleVAE((200, n, n), 256, 64).to(
        device
    )  # 特徴量, 分類クラス数, 中間のチャンネル数
    optimizer = torch.optim.Adam(lr=1e-6, params=vae.parameters())
    train(
        vae,
        optimizer,
        train_dataset,
        val_dataset,
        epochs=100,
        transform=normalize_images,
    )
    evaluate(vae, test_dataset, normalize_images, device)


main()
