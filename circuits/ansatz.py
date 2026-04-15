"""
ansatz.py — Variational quantum circuit architectures for QCL experiments.

Three ansätze with TorchLayer-compatible signatures (inputs first, weights second):
  - strongly_entangling : Strongly Entangling Layers (SEL), 24 parameters for n=4, L=2.
  - basic_entangler     : Basic Entangler Layers, 8 parameters for n=4, L=2.
  - ttn                 : Tree Tensor Network, 6 parameters for n=4.

Each circuit returns two PauliZ expectation values (wires 0 and 1) for use
with nn.Linear(2, 2) + CrossEntropyLoss (matches the original HybridQuantumNet).
"""

import numpy as np
import pennylane as qml

ANSATZ_NAMES = ("strongly_entangling", "basic_entangler", "ttn")


def get_weight_shapes(ansatz: str, n_qubits: int = 4, n_layers: int = 2) -> dict:
    """Return weight_shapes dict for qml.qnn.TorchLayer."""
    if ansatz == "strongly_entangling":
        return {"weights": (n_layers, n_qubits, 3)}
    elif ansatz == "basic_entangler":
        return {"weights": (n_layers, n_qubits)}
    elif ansatz == "ttn":
        return {"weights": (2 * (n_qubits - 1),)}
    raise ValueError(f"Unknown ansatz: {ansatz}")


def get_param_shape(ansatz: str, n_qubits: int = 4, n_layers: int = 2) -> tuple:
    """Return raw parameter shape (for numpy initialization)."""
    return get_weight_shapes(ansatz, n_qubits, n_layers)["weights"]


def get_param_count(ansatz: str, n_qubits: int = 4, n_layers: int = 2) -> int:
    shape = get_param_shape(ansatz, n_qubits, n_layers)
    return int(np.prod(shape))


def _apply_ttn(weights, n_qubits: int = 4):
    """
    Tree Tensor Network structure for n_qubits=4.

    Level 1 (leaf pairs):
        [0,1] — RY(w0) on q0, RY(w1) on q1, CNOT(0->1)
        [2,3] — RY(w2) on q2, RY(w3) on q3, CNOT(2->3)
    Level 2 (root merge):
        [1,3] — RY(w4) on q1, RY(w5) on q3, CNOT(1->3)
    """
    if n_qubits != 4:
        raise NotImplementedError("TTN implemented for n_qubits=4 only.")
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RY(weights[4], wires=1)
    qml.RY(weights[5], wires=3)
    qml.CNOT(wires=[1, 3])


def build_circuit(
    ansatz: str,
    n_qubits: int = 4,
    n_layers: int = 2,
    noise_ops: list | None = None,
    backend: str = "default.qubit",
    diff_method: str = "backprop",
):
    """
    Build a PennyLane QNode with TorchLayer-compatible signature.

    Signature: circuit(inputs, weights)
      - inputs  : shape (n_qubits,), feature vector in [0, pi]
      - weights : variational parameters per get_param_shape()

    Returns [expval(PauliZ(0)), expval(PauliZ(1))] for CrossEntropyLoss
    via nn.Linear(2, 2).

    Parameters
    ----------
    ansatz : str
        One of ANSATZ_NAMES.
    n_qubits : int
        Number of qubits (default 4).
    n_layers : int
        Number of variational layers for SEL and Basic Entangler.
    noise_ops : list or None
        List of (pennylane_fn, wires, kwargs) inserted after the variational
        block for density-matrix noise simulation.
    backend : str
        "default.qubit" for ideal, "default.mixed" for noisy simulation.
    diff_method : str
        "backprop" for both backends (most compatible with TorchLayer).

    Returns
    -------
    qnode : callable
    """
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(inputs, weights):
        # Ry angle embedding — same as original AngleEmbedding with rotation="Y"
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        if ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif ansatz == "basic_entangler":
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        elif ansatz == "ttn":
            _apply_ttn(weights, n_qubits)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        # Noise channels (only active when noise_ops is not None)
        if noise_ops:
            for noise_fn, wires, kwargs in noise_ops:
                for w in wires:
                    noise_fn(wires=w, **kwargs)

        # Two measurements for CrossEntropyLoss via nn.Linear(2, 2)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    return circuit
