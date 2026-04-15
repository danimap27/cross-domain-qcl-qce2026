"""
ansatz.py — Variational quantum circuit architectures for QCL experiments.

Three ansätze are implemented:
  - strongly_entangling : Strongly Entangling Layers (SEL), high expressibility.
  - basic_entangler     : Basic Entangler Layers, low parametrization.
  - ttn                 : Tree Tensor Network, hierarchical structure.

Each circuit applies Ry angle embedding followed by the variational block.
"""

import numpy as np
import pennylane as qml
from typing import Literal


ANSATZ_NAMES = ("strongly_entangling", "basic_entangler", "ttn")


def get_param_shape(ansatz: str, n_qubits: int = 4, n_layers: int = 2) -> tuple:
    """Return the parameter shape for a given ansatz."""
    if ansatz == "strongly_entangling":
        return (n_layers, n_qubits, 3)
    elif ansatz == "basic_entangler":
        return (n_layers, n_qubits)
    elif ansatz == "ttn":
        # 4 qubits: 3 pairs at layer 1 (2 RY per pair) + 1 CNOT merge = 7 params
        # Exact count: (n_qubits - 1) * 2 parametrized rotations in a binary tree
        return (2 * (n_qubits - 1),)
    raise ValueError(f"Unknown ansatz: {ansatz}")


def get_param_count(ansatz: str, n_qubits: int = 4, n_layers: int = 2) -> int:
    shape = get_param_shape(ansatz, n_qubits, n_layers)
    return int(np.prod(shape))


def _apply_ttn(params: np.ndarray, n_qubits: int = 4):
    """
    Tree Tensor Network structure for n_qubits=4.

    Level 1 (leaf pairs):
        [0,1] — RY(p0) on q0, RY(p1) on q1, CNOT(0->1)
        [2,3] — RY(p2) on q2, RY(p3) on q3, CNOT(2->3)
    Level 2 (root merge):
        [1,3] — RY(p4) on q1, RY(p5) on q3, CNOT(1->3)
    """
    if n_qubits != 4:
        raise NotImplementedError("TTN implemented for n_qubits=4 only.")
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=2)
    qml.RY(params[3], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RY(params[4], wires=1)
    qml.RY(params[5], wires=3)
    qml.CNOT(wires=[1, 3])


def build_circuit(
    ansatz: str,
    n_qubits: int = 4,
    n_layers: int = 2,
    noise_ops: list | None = None,
    backend: str = "default.qubit",
    diff_method: str = "adjoint",
):
    """
    Build a PennyLane QNode for the given ansatz.

    Parameters
    ----------
    ansatz : str
        One of "strongly_entangling", "basic_entangler", "ttn".
    n_qubits : int
        Number of qubits (default 4).
    n_layers : int
        Number of variational layers (ignored for TTN).
    noise_ops : list or None
        List of (noise_fn, wires, kwargs) tuples to insert after each gate layer.
        Used for noisy simulations with default.mixed.
    backend : str
        PennyLane device string. Use "default.mixed" for noisy simulations.
    diff_method : str
        Differentiation method ("adjoint" for ideal, "backprop" for noisy).

    Returns
    -------
    qnode : callable
        A QNode with signature qnode(params, x) -> float.
    """
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params, x):
        # Angle embedding: Ry rotation encoding
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")

        if ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        elif ansatz == "basic_entangler":
            qml.BasicEntanglerLayers(params, wires=range(n_qubits))
        elif ansatz == "ttn":
            _apply_ttn(params, n_qubits)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        # Insert noise operators after the variational block (noisy mode)
        if noise_ops:
            for noise_fn, wires, kwargs in noise_ops:
                for w in wires:
                    noise_fn(wires=w, **kwargs)

        return qml.expval(qml.PauliZ(0))

    return circuit
