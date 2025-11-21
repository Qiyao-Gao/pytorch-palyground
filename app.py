import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    history: List[float]
    accuracies: List[float]
    model: nn.Module
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_pred: np.ndarray
    snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def make_dataset(
    name: str,
    n_samples: int,
    noise: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(random_state)
    if name == "rings":
        X, y = datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.4, random_state=random_state
        )
        n_classes = 2
    elif name == "xor":
        centers = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        base = n_samples // 4
        remainder = n_samples - base * 4
        counts = np.full(4, base)
        counts[:remainder] += 1  # spread leftover samples evenly across quadrants

        jitter = np.concatenate(
            [rng.normal(scale=0.4 + noise, size=(count, 2)) for count in counts], axis=0
        )
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


def build_mlp(
    input_dim: int,
    hidden_layers: int,
    hidden_units: int,
    activation_name: str,
    output_dim: int,
) -> nn.Module:
    activation = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
    }[activation_name]

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(prev_dim, hidden_units))
        layers.append(activation())
        prev_dim = hidden_units

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    hidden_layers: int,
    hidden_units: int,
    activation: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    test_split: float,
    device: torch.device,
    on_epoch: Optional[Callable[[int, List[float], List[float], nn.Module], None]] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> TrainingResult:
    if any(arr is None for arr in [X_train, y_train, X_test, y_test]):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_mlp(input_dim=X.shape[1], hidden_layers=hidden_layers, hidden_units=hidden_units, activation_name=activation, output_dim=n_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[float] = []
    accuracies: List[float] = []
    snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for epoch_idx in range(epochs):
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
        history.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        accuracies.append(correct / max(total, 1))

        if on_epoch is not None:
            on_epoch(epoch_idx, history, accuracies, model)

        if epoch_idx % max(1, epochs // 12) == 0 or epoch_idx == epochs - 1:
            snapshots.append(build_decision_boundary(model, X, device, grid_size=120))

    grid_x, grid_y, grid_pred = build_decision_boundary(model, X, device)
    return TrainingResult(
        history=history,
        accuracies=accuracies,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_pred=grid_pred,
        snapshots=snapshots,
    )


def build_decision_boundary(
    model: nn.Module, X: np.ndarray, device: torch.device, grid_size: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(grid).to(device)).argmax(dim=1).cpu().numpy()
    return xx, yy, preds.reshape(xx.shape)


def plot_decision_boundary(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_pred: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    n_classes: int,
    show_test: bool,
) -> plt.Figure:
    colors = plt.cm.get_cmap("tab10", n_classes)
    cmap = ListedColormap(colors(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(grid_x, grid_y, grid_pred, alpha=0.3, cmap=cmap, levels=n_classes)
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor="k", label="train", s=25
    )
    if show_test and X_test is not None and y_test is not None:
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cmap,
            edgecolor="white",
            label="test",
            s=30,
            marker="^",
        )
    ax.legend(loc="upper right")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision boundary")
    fig.tight_layout()
    return fig


def plot_training_curves(history: List[float], accuracies: List[float]) -> plt.Figure:
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, accuracies, marker="o", color="green")
    axes[1].set_title("Test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="PyTorch Playground", layout="wide")
    st.title("PyTorch Playground")
    st.markdown(
        "Experiment with a small multilayer perceptron (MLP) built with **PyTorch**.\n"
        "Tune the architecture, optimizer, and dataset noise to see how the decision boundary changes."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.header("Data")
    dataset = st.sidebar.selectbox(
        "Dataset", ["rings", "xor", "spiral", "two-sides"], index=0, format_func=str.title
    )
    n_samples = st.sidebar.slider("Samples", min_value=200, max_value=2000, value=800, step=50)
    noise = st.sidebar.slider("Noise", min_value=0.0, max_value=0.6, value=0.2, step=0.02)
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    st.sidebar.header("Model")
    hidden_layers = st.sidebar.slider("Hidden layers", 0, 4, 2)
    hidden_units = st.sidebar.slider("Hidden units", 2, 50, 8)
    activation = st.sidebar.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"], index=0)

    st.sidebar.header("Training")
    lr = st.sidebar.number_input("Learning rate", min_value=1e-4, max_value=1.0, value=0.01, step=0.001, format="%.4f")
    weight_decay = st.sidebar.number_input("Weight decay", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    batch_size = st.sidebar.slider("Batch size", 8, 256, 64, step=8)
    epochs = st.sidebar.slider("Epochs", 1, 200, 60, step=1)
    test_split = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
    show_test = st.sidebar.checkbox("Show test data", value=True)

    st.sidebar.info(f"Using device: {device}")

    if "result" not in st.session_state:
        st.session_state.result = None

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("How to use")
        st.markdown(
            "1. Choose a toy dataset and adjust its noise.\n"
            "2. Configure the number of layers, units, and activation.\n"
            "3. Set the optimizer hyperparameters.\n"
            "4. Press **Train network** to see the new decision boundary."
        )
        if st.button("Train network", type="primary"):
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            X, y, n_classes = make_dataset(dataset, n_samples, noise, random_state)
            boundary_placeholder = st.empty()
            metrics_placeholder = st.empty()
            progress = st.progress(0.0, text="Training model...")

            def update_boundary(epoch_idx: int, history: List[float], acc: List[float], model: nn.Module):
                """Stream live decision boundary + curves into placeholders during training."""
                grid_x, grid_y, grid_pred = build_decision_boundary(model, X, device, grid_size=120)
                boundary_container = boundary_placeholder.container()
                boundary_container.caption(
                    f"Epoch {epoch_idx + 1}/{epochs} – loss {history[-1]:.4f}, test acc {acc[-1]*100:.1f}%"
                )
                boundary_fig = plot_decision_boundary(
                    grid_x,
                    grid_y,
                    grid_pred,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_classes=n_classes,
                    show_test=show_test,
                )
                boundary_container.pyplot(boundary_fig)
                plt.close(boundary_fig)

                metrics_fig = plot_training_curves(history, acc)
                metrics_placeholder.pyplot(metrics_fig)
                plt.close(metrics_fig)

                progress.progress(min((epoch_idx + 1) / epochs, 1.0))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, stratify=y
            )
            st.session_state.result = train_model(
                X=X,
                y=y,
                n_classes=n_classes,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                hidden_layers=hidden_layers,
                hidden_units=hidden_units,
                activation=activation,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                test_split=test_split,
                device=device,
                on_epoch=update_boundary,
            )
            progress.empty()
            st.session_state.n_classes = n_classes
        st.divider()
        st.markdown("This playground runs everything locally in your browser session using CPU.")

    with col2:
        if st.session_state.result is None:
            st.info("Configure the model on the left and click **Train network** to begin.")
        else:
            result: TrainingResult = st.session_state.result
            st.subheader("Decision boundary")
            boundary_fig = plot_decision_boundary(
                result.grid_x,
                result.grid_y,
                result.grid_pred,
                result.X_train,
                result.y_train,
                result.X_test,
                result.y_test,
                st.session_state.n_classes,
                show_test,
            )
            st.pyplot(boundary_fig)
            plt.close(boundary_fig)

            st.subheader("Training curves")
            metrics_fig = plot_training_curves(result.history, result.accuracies)
            st.pyplot(metrics_fig)
            plt.close(metrics_fig)

            st.metric("Final test accuracy", f"{result.accuracies[-1]*100:.1f}%")
            st.caption(
                "Test accuracy is computed on a hold-out split of the generated dataset after every epoch."
            )


if __name__ == "__main__":
    main()
