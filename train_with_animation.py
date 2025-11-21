"""
Live training visualization for a small PyTorch MLP on 2D toy datasets.

This script reuses the dataset/model helpers from the playground app but adds
matplotlib.animation.FuncAnimation to show the decision boundary evolving
alongside the training loss curve in real time.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    dataset: str = "spiral"
    n_samples: int = 800
    noise: float = 0.2
    random_state: int = 42
    hidden_layers: int = 2
    hidden_units: int = 8
    activation: str = "ReLU"
    lr: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 60
    test_split: float = 0.2


def make_dataset(name: str, n_samples: int, noise: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(random_state)
    if name == "rings":
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=random_state)
        n_classes = 2
    elif name == "xor":
        centers = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        base = n_samples // 4
        remainder = n_samples - base * 4
        counts = np.full(4, base)
        counts[:remainder] += 1

        jitter = np.concatenate([rng.normal(scale=0.4 + noise, size=(count, 2)) for count in counts], axis=0)
        X = np.repeat(centers, counts, axis=0) + jitter
        y = np.array([0, 1, 1, 0], dtype=np.int64).repeat(counts)
        n_classes = 2
    elif name == "spiral":
        n_samples_per_class = n_samples // 2
        theta = np.linspace(0, 3 * math.pi, n_samples_per_class)
        r = np.linspace(0.2, 5, n_samples_per_class)
        x1 = np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)
        x2 = np.stack((r * np.cos(theta + math.pi), r * np.sin(theta + math.pi)), axis=1)
        X = np.concatenate([x1, x2], axis=0)
        X += rng.normal(scale=noise, size=X.shape)
        y = np.concatenate([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])
        n_classes = 2
    else:  # two-sides / blobs
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=[(-3, 0), (3, 0)],
            cluster_std=1.2 + noise,
            random_state=random_state,
        )
        n_classes = 2

    return X.astype(np.float32), y.astype(np.int64), n_classes


def build_mlp(input_dim: int, hidden_layers: int, hidden_units: int, activation_name: str, output_dim: int) -> nn.Module:
    activation = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}[activation_name]
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(prev_dim, hidden_units))
        layers.append(activation())
        prev_dim = hidden_units
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / max(total, 1)


def main() -> None:
    cfg = TrainingConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation using the existing generation logic.
    X, y, n_classes = make_dataset(cfg.dataset, cfg.n_samples, cfg.noise, cfg.random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_split, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = build_mlp(
        input_dim=X.shape[1],
        hidden_layers=cfg.hidden_layers,
        hidden_units=cfg.hidden_units,
        activation_name=cfg.activation,
        output_dim=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Pre-compute grid used for the decision boundary visualization.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    grid_size = 200
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    grid_tensor = torch.from_numpy(grid_points).to(device)

    def predict_grid() -> np.ndarray:
        """Run the model over the mesh grid and return class predictions."""
        model.eval()
        with torch.no_grad():
            preds = model(grid_tensor).argmax(dim=1).cpu().numpy()
        return preds.reshape(grid_x.shape)

    # --- Matplotlib setup -------------------------------------------------
    fig, (ax_data, ax_curve) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle("PyTorch MLP training (live)")

    cmap = plt.cm.get_cmap("tab10", n_classes)

    # Left subplot: decision regions + points.
    boundary_im = ax_data.imshow(
        predict_grid(),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=cmap,
        alpha=0.3,
        interpolation="nearest",
        aspect="auto",
    )
    train_scatter = ax_data.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors="k", s=25, label="train")
    test_scatter = ax_data.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, edgecolors="white", s=30, marker="^", label="test")
    ax_data.set_xlim(x_min, x_max)
    ax_data.set_ylim(y_min, y_max)
    ax_data.set_xlabel("Feature 1")
    ax_data.set_ylabel("Feature 2")
    ax_data.set_title("Decision boundary")
    ax_data.legend(loc="upper right")

    # Right subplot: training loss + accuracy curves.
    loss_line, = ax_curve.plot([], [], label="loss")
    acc_line, = ax_curve.plot([], [], label="test acc")
    ax_curve.set_xlim(1, cfg.epochs)
    ax_curve.set_ylim(0, 1.2)
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Loss / Accuracy")
    ax_curve.set_title("Training curves")
    ax_curve.legend()

    losses: List[float] = []
    accuracies: List[float] = []

    def train_one_epoch() -> Tuple[float, float]:
        """Run one epoch of training and return (avg_loss, test_accuracy)."""
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        test_acc = evaluate_accuracy(model, test_loader, device)
        return avg_loss, test_acc

    def init_animation():
        """Initialize artists for FuncAnimation (required for blitting)."""
        loss_line.set_data([], [])
        acc_line.set_data([], [])
        boundary_im.set_data(predict_grid())
        return boundary_im, loss_line, acc_line

    def update(frame_idx: int):
        """
        Run one epoch of training, then refresh the plots.

        This function is called by FuncAnimation every frame, creating the
        perception of a live-updating decision boundary and learning curves.
        """
        avg_loss, test_acc = train_one_epoch()
        losses.append(avg_loss)
        accuracies.append(test_acc)

        # Update decision boundary background.
        boundary_im.set_data(predict_grid())

        # Update curves.
        epochs_axis = np.arange(1, len(losses) + 1)
        loss_line.set_data(epochs_axis, losses)
        acc_line.set_data(epochs_axis, accuracies)
        ax_curve.set_ylim(0, max(1.2, max(losses + accuracies)))

        ax_curve.figure.canvas.draw_idle()
        return boundary_im, loss_line, acc_line

    # Create the animation; interval controls the pause between frames (ms).
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init_animation,
        frames=cfg.epochs,
        interval=200,
        blit=False,
        repeat=False,
    )

    print("Starting training with live visualization...")
    plt.show()
    # Saving the animation (optional): uncomment if you want a GIF/MP4.
    # anim.save("training.gif", writer="pillow", fps=10)


if __name__ == "__main__":
    main()
