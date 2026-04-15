"""
synthetic.py — Synthetic Gaussian dataset generator for QCL pre-training.

Two isotropic Gaussian clusters are generated in R^n_features, separated
by a fixed distance. This domain has no semantic overlap with the target
image tasks (Fashion-MNIST, MNIST), making it a true cross-domain source.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
import torch


def make_synthetic_gaussian(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 2,
    separation: float = 3.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> TensorDataset:
    """
    Generate a binary classification dataset with two Gaussian clusters.

    Class k has mean at +/- (separation/2) along the first feature axis.
    Remaining features are drawn from N(0, noise_std^2).

    Parameters
    ----------
    n_samples : int
        Total number of samples (split equally across classes).
    n_features : int
        Dimensionality of each sample (typically equals n_qubits).
    n_classes : int
        Number of classes (currently only 2 is supported).
    separation : float
        Distance between class centroids along the first axis.
    noise_std : float
        Standard deviation of Gaussian noise on all features.
    seed : int
        Random seed.

    Returns
    -------
    dataset : TensorDataset
        Dataset with (X, y) where X is in [0, pi]^n_features and y in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // n_classes

    X_parts, y_parts = [], []
    for k in range(n_classes):
        center = np.zeros(n_features)
        center[0] = (k - (n_classes - 1) / 2.0) * separation
        X_k = rng.normal(loc=center, scale=noise_std, size=(n_per_class, n_features))
        X_parts.append(X_k)
        y_parts.append(np.full(n_per_class, k, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Scale to [0, pi] for Ry angle embedding
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X = scaler.fit_transform(X).astype(np.float32)

    idx = rng.permutation(len(X))
    return TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
