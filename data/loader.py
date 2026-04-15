"""
loader.py — Dataset loaders for QCL sequential tasks.

Each task is a binary subset of a standard image dataset, reduced to
n_qubits features via PCA and scaled to [0, pi] for Ry angle embedding.

Supported datasets:
  - fashion_mnist : Zalando Fashion-MNIST, 10 classes.
  - mnist         : Handwritten digit MNIST, 10 classes.

The MobileNetV2 feature source uses torchvision to extract 1280-dim
embeddings, which are then reduced to n_features via PCA.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def _get_raw_data(dataset_name: str, train: bool, data_dir: str = "./data/raw") -> Tuple[np.ndarray, np.ndarray]:
    """Download and return raw (X_flat, y) arrays from torchvision."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "fashion_mnist":
        ds = torchvision.datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "mnist":
        ds = torchvision.datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb.view(xb.size(0), -1).numpy())
        y_list.append(yb.numpy())
    return np.vstack(X_list), np.concatenate(y_list)


def _extract_mobilenetv2_features(
    dataset_name: str,
    train: bool,
    classes: list[int],
    data_dir: str = "./data/raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract MobileNetV2 feature embeddings (1280-dim) from raw images."""
    import torchvision
    import torchvision.transforms as T
    import torchvision.models as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize(224),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "fashion_mnist":
        ds = torchvision.datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "mnist":
        ds = torchvision.datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Filter classes
    mask = np.isin(np.array(ds.targets), classes)
    indices = np.where(mask)[0].tolist()
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=2)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)

    feats, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            feats.append(model(xb.to(device)).cpu().numpy())
            labels.append(yb.numpy())

    X = np.vstack(feats)
    y = np.concatenate(labels)

    # Remap class indices to {0, 1}
    for new_idx, cls in enumerate(sorted(classes)):
        y[y == cls] = new_idx
    return X, y


def load_task_pca(
    dataset_name: str,
    classes: list[int],
    n_features: int = 4,
    source: str = "pixel",
    train_ratio: float = 0.8,
    data_dir: str = "./data/raw",
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load a binary task, reduce to n_features via PCA, scale to [0, pi].

    Parameters
    ----------
    dataset_name : str
        "fashion_mnist" or "mnist".
    classes : list of int
        Two class indices to select (e.g., [0, 1]).
    n_features : int
        Number of PCA components (must equal n_qubits).
    source : str
        "pixel" for raw flattened pixels, "mobilenetv2" for CNN features.
    train_ratio : float
        Fraction of data for training when a single split is requested.
    data_dir : str
        Root directory for dataset downloads.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_ds, test_ds : TensorDataset, TensorDataset
    """
    if source == "mobilenetv2":
        X_tr, y_tr = _extract_mobilenetv2_features(dataset_name, True, classes, data_dir)
        X_te, y_te = _extract_mobilenetv2_features(dataset_name, False, classes, data_dir)
    else:
        X_all, y_all = _get_raw_data(dataset_name, train=True, data_dir=data_dir)
        X_te_raw, y_te_raw = _get_raw_data(dataset_name, train=False, data_dir=data_dir)

        mask_tr = np.isin(y_all, classes)
        mask_te = np.isin(y_te_raw, classes)
        X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]
        X_te, y_te = X_te_raw[mask_te], y_te_raw[mask_te]

        # Remap class indices to {0, 1}
        for new_idx, cls in enumerate(sorted(classes)):
            y_tr[y_tr == cls] = new_idx
            y_te[y_te == cls] = new_idx

    # PCA fit on training set, transform both splits
    pca = PCA(n_components=n_features, random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    # Scale to [0, pi] for Ry embedding
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_scaled = scaler.fit_transform(X_tr_pca).astype(np.float32)
    X_te_scaled = scaler.transform(X_te_pca).astype(np.float32)

    y_tr = y_tr.astype(np.int64)
    y_te = y_te.astype(np.int64)

    train_ds = TensorDataset(torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te_scaled), torch.from_numpy(y_te))
    return train_ds, test_ds


# Alias for convenience
load_task = load_task_pca
