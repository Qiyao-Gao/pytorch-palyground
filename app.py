import math
from dataclasses import dataclass
from typing import List, Tuple

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
    X_val: np.ndarray
    y_val: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_pred: np.ndarray


def make_dataset(
    name: str,
    n_samples: int,
    noise: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if name == "moons":
        X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        n_classes = 2
    elif name == "circles":
        X, y = datasets.make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=random_state)
        n_classes = 2
    else:
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=3,
            cluster_std=1.2 + noise,
            random_state=random_state,
        )
        n_classes = 3
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
    val_split: float,
    device: torch.device,
) -> TrainingResult:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_mlp(input_dim=X.shape[1], hidden_layers=hidden_layers, hidden_units=hidden_units, activation_name=activation, output_dim=n_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[float] = []
    accuracies: List[float] = []

    for _ in range(epochs):
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

    grid_x, grid_y, grid_pred = build_decision_boundary(model, X, device)
    return TrainingResult(
        history=history,
        accuracies=accuracies,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_pred=grid_pred,
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
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
):
    colors = plt.cm.get_cmap("tab10", n_classes)
    cmap = ListedColormap(colors(range(n_classes)))
    plt.figure(figsize=(7, 5))
    plt.contourf(grid_x, grid_y, grid_pred, alpha=0.3, cmap=cmap, levels=n_classes)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor="k", label="train", s=25)
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cmap, edgecolor="white", label="val", s=25, marker="^")
    plt.legend(loc="upper right")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision boundary")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


def plot_training_curves(history: List[float], accuracies: List[float]):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, accuracies, marker="o", color="green")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="PyTorch Playground", layout="wide")
    st.title("PyTorch Playground")
    st.markdown(
        "Experiment with a small multilayer perceptron (MLP) built with **PyTorch**.\n"
        "Tune the architecture, optimizer, and dataset noise to see how the decision boundary changes."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.header("Data")
    dataset = st.sidebar.selectbox("Dataset", ["moons", "circles", "blobs"], index=0)
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
    val_split = st.sidebar.slider("Validation split", 0.1, 0.4, 0.2, step=0.05)

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
            with st.spinner("Training model..."):
                st.session_state.result = train_model(
                    X=X,
                    y=y,
                    n_classes=n_classes,
                    hidden_layers=hidden_layers,
                    hidden_units=hidden_units,
                    activation=activation,
                    lr=lr,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    epochs=epochs,
                    val_split=val_split,
                    device=device,
                )
                st.session_state.n_classes = n_classes
        st.divider()
        st.markdown("This playground runs everything locally in your browser session using CPU.")

    with col2:
        if st.session_state.result is None:
            st.info("Configure the model on the left and click **Train network** to begin.")
        else:
            result: TrainingResult = st.session_state.result
            st.subheader("Decision boundary")
            plot_decision_boundary(
                result.grid_x,
                result.grid_y,
                result.grid_pred,
                result.X_train,
                result.y_train,
                result.X_val,
                result.y_val,
                st.session_state.n_classes,
            )

            st.subheader("Training curves")
            plot_training_curves(result.history, result.accuracies)

            st.metric("Final validation accuracy", f"{result.accuracies[-1]*100:.1f}%")
            st.caption(
                "Validation accuracy is computed on a hold-out split of the generated dataset after every epoch."
            )


if __name__ == "__main__":
    main()
