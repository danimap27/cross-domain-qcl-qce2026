"""
trainer.py — Quantum Continual Learning training loop.

Implements the QCL sequential training protocol:
  1. Optional pre-training on a source domain.
  2. Training on Task A; record initial Task A accuracy.
  3. Training on Task B (sequential, no replay buffer).
  4. Evaluation on Task A to compute the forgetting drop.

All parameters are updated via a classical optimizer (Adam) on the
expectation value output of the quantum circuit.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hybrid quantum-classical model
# ---------------------------------------------------------------------------

class HybridQCLModel(nn.Module):
    """
    Hybrid model: a parametrized quantum circuit as a binary classifier.

    The circuit receives a 4-dimensional input (from PCA/synthetic data)
    and returns the expectation value of PauliZ(0), mapped to class
    probabilities via a sigmoid.

    Parameters
    ----------
    circuit : callable
        A PennyLane QNode with signature circuit(params, x) -> float.
    param_shape : tuple
        Shape of the variational parameter tensor.
    init_params : np.ndarray or None
        Initial parameter values. If None, uses random uniform in [-pi, pi].
    """

    def __init__(self, circuit: Callable, param_shape: tuple, init_params=None):
        super().__init__()
        if init_params is not None:
            params = torch.tensor(init_params, dtype=torch.float32)
        else:
            params = (torch.rand(param_shape) * 2 * np.pi) - np.pi
        self.params = nn.Parameter(params)
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)

        Returns
        -------
        logits : torch.Tensor of shape (batch_size,)
            Raw expectation values in [-1, 1].
        """
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, dtype=torch.float32)
        for i in range(batch_size):
            out[i] = self.circuit(self.params, x[i])
        return out


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _train_epoch(model: HybridQCLModel, loader: DataLoader, optimizer, criterion) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        # Map labels {0, 1} -> {-1, +1} to match PauliZ expectation range
        targets = (yb.float() * 2.0 - 1.0)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def _evaluate(model: HybridQCLModel, loader: DataLoader) -> float:
    """Evaluate binary accuracy. Positive expectation -> class 1."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = (logits > 0).long()
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# QCL run config and result
# ---------------------------------------------------------------------------

@dataclass
class QCLRunConfig:
    run_id: str
    ansatz: str
    noise_model: str              # "ideal" or "ibm_heron_r2"
    source: str                   # "scratch", "synthetic_gaussian", "mobilenetv2"
    seed: int
    n_qubits: int = 4
    n_layers: int = 2
    lr: float = 0.05
    epochs: int = 10
    pretrain_epochs: int = 10
    batch_size: int = 32
    data_dir: str = "./data/raw"
    results_dir: str = "./results"
    noise_channels: list = field(default_factory=lambda: ["amplitude_damping", "phase_damping", "depolarizing"])
    machine_id: str = "local"


@dataclass
class QCLResult:
    run_id: str
    ansatz: str
    noise_model: str
    source: str
    seed: int
    acc_source: float
    acc_a_initial: float
    acc_b_final: float
    acc_a_final: float
    forgetting_drop: float
    train_time_source_s: float
    train_time_a_s: float
    train_time_b_s: float
    status: str = "completed"
    error: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Main QCL training function
# ---------------------------------------------------------------------------

def run_qcl(cfg: QCLRunConfig) -> QCLResult:
    """
    Execute a full QCL sequential training run.

    Steps:
      1. Build quantum circuit (ideal or noisy).
      2. Pre-train on source domain (optional).
      3. Train on Task A (Fashion-MNIST classes 0 vs 1).
      4. Train on Task B (MNIST classes 2 vs 3).
      5. Evaluate forgetting on Task A.

    Parameters
    ----------
    cfg : QCLRunConfig
        Full run configuration.

    Returns
    -------
    result : QCLResult
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Build circuit ---
    from circuits.ansatz import build_circuit, get_param_shape
    from circuits.noise import build_noise_operators, IBM_HERON_R2

    is_noisy = cfg.noise_model != "ideal"
    if is_noisy:
        backend = "default.mixed"
        diff_method = "backprop"
        noise_ops = build_noise_operators(
            n_qubits=cfg.n_qubits,
            channels=cfg.noise_channels,
        )
    else:
        backend = "default.qubit"
        diff_method = "adjoint"
        noise_ops = None

    circuit = build_circuit(
        ansatz=cfg.ansatz,
        n_qubits=cfg.n_qubits,
        n_layers=cfg.n_layers,
        noise_ops=noise_ops,
        backend=backend,
        diff_method=diff_method,
    )
    param_shape = get_param_shape(cfg.ansatz, cfg.n_qubits, cfg.n_layers)

    # --- Load data ---
    from data.loader import load_task_pca
    from data.synthetic import make_synthetic_gaussian

    task_a_train, task_a_test = load_task_pca(
        "fashion_mnist", classes=[0, 1], n_features=cfg.n_qubits,
        source="pixel", data_dir=cfg.data_dir, seed=cfg.seed,
    )
    task_b_train, task_b_test = load_task_pca(
        "mnist", classes=[2, 3], n_features=cfg.n_qubits,
        source="pixel", data_dir=cfg.data_dir, seed=cfg.seed,
    )

    loader_a_train = DataLoader(task_a_train, batch_size=cfg.batch_size, shuffle=True)
    loader_a_test = DataLoader(task_a_test, batch_size=256)
    loader_b_train = DataLoader(task_b_train, batch_size=cfg.batch_size, shuffle=True)
    loader_b_test = DataLoader(task_b_test, batch_size=256)

    # --- Initialize model ---
    model = HybridQCLModel(circuit, param_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    acc_source = 0.0
    train_time_source_s = 0.0

    # --- Phase 0: Pre-training ---
    if cfg.source == "synthetic_gaussian":
        source_ds = make_synthetic_gaussian(
            n_samples=1000, n_features=cfg.n_qubits,
            n_classes=2, seed=cfg.seed,
        )
        source_loader = DataLoader(source_ds, batch_size=cfg.batch_size, shuffle=True)
        source_test = DataLoader(source_ds, batch_size=256)

        t0 = time.time()
        for _ in range(cfg.pretrain_epochs):
            _train_epoch(model, source_loader, optimizer, criterion)
        train_time_source_s = time.time() - t0
        acc_source = _evaluate(model, source_test)
        logger.info(f"[{cfg.run_id}] Pre-train (synthetic): acc={acc_source:.4f}, t={train_time_source_s:.1f}s")

    elif cfg.source == "mobilenetv2":
        src_train, src_test_ds = load_task_pca(
            "fashion_mnist", classes=[0, 1], n_features=cfg.n_qubits,
            source="mobilenetv2", data_dir=cfg.data_dir, seed=cfg.seed,
        )
        source_loader = DataLoader(src_train, batch_size=cfg.batch_size, shuffle=True)
        source_test = DataLoader(src_test_ds, batch_size=256)

        t0 = time.time()
        for _ in range(cfg.pretrain_epochs):
            _train_epoch(model, source_loader, optimizer, criterion)
        train_time_source_s = time.time() - t0
        acc_source = _evaluate(model, source_test)
        logger.info(f"[{cfg.run_id}] Pre-train (mobilenetv2): acc={acc_source:.4f}, t={train_time_source_s:.1f}s")

    # --- Phase 1: Task A ---
    t0 = time.time()
    for ep in range(cfg.epochs):
        loss = _train_epoch(model, loader_a_train, optimizer, criterion)
        if (ep + 1) % 5 == 0:
            logger.debug(f"[{cfg.run_id}] Task A ep {ep+1}: loss={loss:.4f}")
    train_time_a_s = time.time() - t0
    acc_a_initial = _evaluate(model, loader_a_test)
    logger.info(f"[{cfg.run_id}] Task A: acc_initial={acc_a_initial:.4f}, t={train_time_a_s:.1f}s")

    # --- Phase 2: Task B (sequential, no replay) ---
    t0 = time.time()
    for ep in range(cfg.epochs):
        loss = _train_epoch(model, loader_b_train, optimizer, criterion)
        if (ep + 1) % 5 == 0:
            logger.debug(f"[{cfg.run_id}] Task B ep {ep+1}: loss={loss:.4f}")
    train_time_b_s = time.time() - t0
    acc_b_final = _evaluate(model, loader_b_test)

    # --- Forgetting evaluation ---
    acc_a_final = _evaluate(model, loader_a_test)
    forgetting_drop = acc_a_initial - acc_a_final

    logger.info(
        f"[{cfg.run_id}] Task B: acc_b={acc_b_final:.4f} | "
        f"acc_a_final={acc_a_final:.4f} | drop={forgetting_drop:.4f}, t={train_time_b_s:.1f}s"
    )

    return QCLResult(
        run_id=cfg.run_id,
        ansatz=cfg.ansatz,
        noise_model=cfg.noise_model,
        source=cfg.source,
        seed=cfg.seed,
        acc_source=acc_source,
        acc_a_initial=acc_a_initial,
        acc_b_final=acc_b_final,
        acc_a_final=acc_a_final,
        forgetting_drop=forgetting_drop,
        train_time_source_s=train_time_source_s,
        train_time_a_s=train_time_a_s,
        train_time_b_s=train_time_b_s,
    )
