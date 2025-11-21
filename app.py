import math
import multiprocessing as mp
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
    model: Optional[nn.Module]
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
    on_epoch: Optional[
        Callable[[int, List[float], List[float], nn.Module, np.ndarray], None]
    ] = None,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    grid_tensor: Optional[torch.Tensor] = None,
) -> TrainingResult:
    if any(arr is None for arr in [X_train, y_train, X_test, y_test]):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_mlp(
        input_dim=X.shape[1],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation_name=activation,
        output_dim=n_classes,
    )
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

        if on_epoch is not None and grid_tensor is not None:
            with torch.no_grad():
                grid_pred = (
                    model(grid_tensor.to(device))
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            on_epoch(epoch_idx, history, accuracies, model, grid_pred)

        if epoch_idx % max(1, epochs // 12) == 0 or epoch_idx == epochs - 1:
            snapshots.append(build_decision_boundary(model, X, device, grid_size=240))

    grid_x, grid_y, grid_pred = build_decision_boundary(model, X, device, grid_size=350)
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
    model: nn.Module, X: np.ndarray, device: torch.device, grid_size: int = 300
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


def make_grid(X: np.ndarray, grid_size: int = 350) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return xx, yy, grid


def init_boundary_plot(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_pred: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    n_classes: int,
    show_test: bool,
) -> Tuple[plt.Figure, plt.Axes, any]:
    # Use a crisp, discrete palette for clearer decision regions
    discrete_colors = plt.get_cmap("Set1", max(n_classes, 3))
    cmap = ListedColormap(discrete_colors(range(max(n_classes, 3))))
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(
        grid_x,
        grid_y,
        grid_pred,
        cmap=cmap,
        shading="nearest",
        rasterized=True,
        vmin=-0.5,
        vmax=n_classes - 0.5,
    )
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cmap,
        edgecolor="k",
        linewidth=0.5,
        label="train",
        s=25,
    )
    if show_test and X_test is not None and y_test is not None:
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cmap,
            edgecolor="white",
            linewidth=0.6,
            label="test",
            s=30,
            marker="^",
        )
    ax.legend(loc="upper right")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision boundary")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax, mesh


def update_boundary_plot(mesh: any, grid_pred: np.ndarray) -> None:
    # Update only the mesh colors for smoother animation
    mesh.set_array(grid_pred.ravel())
    mesh.changed()


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
    fig, ax, _ = init_boundary_plot(
        grid_x, grid_y, grid_pred, X_train, y_train, X_test, y_test, n_classes, show_test
    )
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


def init_training_curves() -> Tuple[plt.Figure, List[plt.Axes], plt.Line2D, plt.Line2D]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    loss_line, = axes[0].plot([], [], marker="o")
    acc_line, = axes[1].plot([], [], marker="o", color="green")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    return fig, axes, loss_line, acc_line


def update_training_curves(
    axes: List[plt.Axes],
    loss_line: plt.Line2D,
    acc_line: plt.Line2D,
    history: List[float],
    accuracies: List[float],
) -> None:
    epochs = np.arange(1, len(history) + 1)
    loss_line.set_data(epochs, history)
    acc_line.set_data(epochs, accuracies)
    for ax in axes:
        ax.relim()
        ax.autoscale_view()


def remote_train_worker(conn, args: dict) -> None:
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    def send_update(epoch_idx: int, history: List[float], acc: List[float], model: nn.Module, grid_pred: np.ndarray):
        conn.send(
            {
                "type": "update",
                "epoch": epoch_idx,
                "history": history,
                "accuracies": acc,
                "grid_pred": grid_pred,
            }
        )

    result = train_model(
        X=args["X"],
        y=args["y"],
        n_classes=args["n_classes"],
        hidden_layers=args["hidden_layers"],
        hidden_units=args["hidden_units"],
        activation=args["activation"],
        lr=args["lr"],
        weight_decay=args["weight_decay"],
        batch_size=args["batch_size"],
        epochs=args["epochs"],
        test_split=args["test_split"],
        device=args["device"],
        on_epoch=send_update,
        X_train=args["X_train"],
        y_train=args["y_train"],
        X_test=args["X_test"],
        y_test=args["y_test"],
        grid_tensor=torch.from_numpy(args["grid_points"]).to(args["device"]),
    )

    conn.send(
        {
            "type": "done",
            "history": result.history,
            "accuracies": result.accuracies,
            "grid_pred": result.grid_pred.ravel(),
        }
    )
    conn.close()


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
    remote_mode = st.sidebar.checkbox("Remote GPU worker (experimental)", value=False)

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

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42, stratify=y
            )

            grid_x, grid_y, grid_points = make_grid(X, grid_size=350)
            grid_tensor = torch.from_numpy(grid_points).to(device)

            # Reusable, non-recreated figures for smoother updates
            boundary_fig = None
            boundary_mesh = None
            metrics_fig = None
            metrics_axes: List[plt.Axes] = []
            loss_line: Optional[plt.Line2D] = None
            acc_line: Optional[plt.Line2D] = None

            def update_boundary(epoch_idx: int, history: List[float], acc: List[float], model: nn.Module, grid_pred_flat: np.ndarray):
                """Stream live decision boundary + curves into placeholders during training."""
                nonlocal boundary_fig, boundary_mesh, metrics_fig, metrics_axes, loss_line, acc_line
                grid_pred = grid_pred_flat.reshape(grid_x.shape)
                if boundary_fig is None:
                    boundary_fig, _, boundary_mesh = init_boundary_plot(
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
                else:
                    update_boundary_plot(boundary_mesh, grid_pred)

                boundary_placeholder.pyplot(boundary_fig)
                boundary_placeholder.caption(
                    f"Epoch {epoch_idx + 1}/{epochs} – loss {history[-1]:.4f}, test acc {acc[-1]*100:.1f}%"
                )

                if metrics_fig is None:
                    metrics_fig, metrics_axes, loss_line, acc_line = init_training_curves()
                update_training_curves(metrics_axes, loss_line, acc_line, history, acc)
                metrics_placeholder.pyplot(metrics_fig)

                progress.progress(min((epoch_idx + 1) / epochs, 1.0))

            if remote_mode:
                parent_conn, child_conn = mp.Pipe()
                args = {
                    "X": X,
                    "y": y,
                    "n_classes": n_classes,
                    "hidden_layers": hidden_layers,
                    "hidden_units": hidden_units,
                    "activation": activation,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "test_split": test_split,
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "grid_points": grid_points,
                    "seed": random_state,
                }
                worker = mp.Process(target=remote_train_worker, args=(child_conn, args))
                worker.start()

                final_history: List[float] = []
                final_acc: List[float] = []
                final_grid_pred = None

                while True:
                    message = parent_conn.recv()
                    if message["type"] == "update":
                        update_boundary(
                            message["epoch"],
                            message["history"],
                            message["accuracies"],
                            None,
                            message["grid_pred"],
                        )
                        final_history = message["history"]
                        final_acc = message["accuracies"]
                    elif message["type"] == "done":
                        final_history = message["history"]
                        final_acc = message["accuracies"]
                        final_grid_pred = message["grid_pred"].reshape(grid_x.shape)
                        break
                worker.join()

                st.session_state.result = TrainingResult(
                    history=final_history,
                    accuracies=final_acc,
                    model=None,  # model resides on the worker
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    grid_pred=final_grid_pred,
                    snapshots=[],
                )
            else:
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
                    grid_tensor=grid_tensor,
                )
            progress.empty()
            st.session_state.n_classes = n_classes
        st.divider()
        st.markdown(
            "This playground runs locally. If a CUDA GPU is available, training automatically uses it; "
            "enable the remote worker option when the UI runs on a node without direct GPU access."
        )

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
