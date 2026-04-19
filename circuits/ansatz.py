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


def _get_noise_layer(noise_params: dict):
    """Helper to pre-calculate noise gammas and return local closure channels."""
    import math
    p1q = noise_params.get("p1q", 0.0002)
    p2q = noise_params.get("p2q", 0.005)
    T1_us = noise_params.get("T1_us", 250)
    T2_us = noise_params.get("T2_us", 150)
    t1q_ns = noise_params.get("t1q_ns", 32)
    t2q_ns = noise_params.get("t2q_ns", 68)

    T1_ns = T1_us * 1000
    T2_ns = T2_us * 1000

    gamma_1q = 1 - math.exp(-t1q_ns / T1_ns) if T1_ns > 0 else 0
    gamma_2q = 1 - math.exp(-t2q_ns / T1_ns) if T1_ns > 0 else 0
    T_phi_inv = (1.0 / T2_ns - 1.0 / (2 * T1_ns)) if T2_ns > 0 else 0
    gphi_1q = 1 - math.exp(-t1q_ns * T_phi_inv) if T_phi_inv > 0 else 0
    gphi_2q = 1 - math.exp(-t2q_ns * T_phi_inv) if T_phi_inv > 0 else 0

    noise_channels = ["amplitude_damping", "phase_damping", "depolarizing"]

    def _noise_1q(wire):
        if "amplitude_damping" in noise_channels:
            qml.AmplitudeDamping(gamma_1q, wires=wire)
        if "phase_damping" in noise_channels:
            qml.PhaseDamping(gphi_1q, wires=wire)
        if "depolarizing" in noise_channels:
            qml.DepolarizingChannel(p1q, wires=wire)

    def _noise_2q(w0, w1):
        if "amplitude_damping" in noise_channels:
            qml.AmplitudeDamping(gamma_2q, wires=w0)
            qml.AmplitudeDamping(gamma_2q, wires=w1)
        if "phase_damping" in noise_channels:
            qml.PhaseDamping(gphi_2q, wires=w0)
            qml.PhaseDamping(gphi_2q, wires=w1)
        if "depolarizing" in noise_channels:
            qml.DepolarizingChannel(p2q, wires=w0)
            qml.DepolarizingChannel(p2q, wires=w1)

    return _noise_1q, _noise_2q


def _apply_strongly_entangling_with_noise(weights, n_qubits, n_layers, noise_params):
    """SEL with integrated noise."""
    _noise_1q, _noise_2q = _get_noise_layer(noise_params)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
            _noise_1q(i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            _noise_2q(i, i + 1)
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
            _noise_2q(n_qubits - 1, 0)


def _apply_basic_entangler_with_noise(weights, n_qubits, n_layers, noise_params):
    """Basic Entangler with integrated noise."""
    _noise_1q, _noise_2q = _get_noise_layer(noise_params)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i], wires=i)
            _noise_1q(i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            _noise_2q(i, i + 1)
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
            _noise_2q(n_qubits - 1, 0)


def _apply_ttn_with_noise(weights, n_qubits, noise_params):
    """TTN with integrated noise (n_qubits=4)."""
    if n_qubits != 4:
        raise NotImplementedError("TTN implemented for n_qubits=4 only.")
    _noise_1q, _noise_2q = _get_noise_layer(noise_params)

    # Level 1
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    _noise_1q(0); _noise_1q(1)
    qml.CNOT(wires=[0, 1]); _noise_2q(0, 1)

    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    _noise_1q(2); _noise_1q(3)
    qml.CNOT(wires=[2, 3]); _noise_2q(2, 3)

    # Level 2
    qml.RY(weights[4], wires=1)
    qml.RY(weights[5], wires=3)
    _noise_1q(1); _noise_1q(3)
    qml.CNOT(wires=[1, 3]); _noise_2q(1, 3)


def _apply_ttn(weights, n_qubits):
    """Ideal TTN (n_qubits=4)."""
    # Level 1
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])

    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CNOT(wires=[2, 3])

    # Level 2
    qml.RY(weights[4], wires=1)
    qml.RY(weights[5], wires=3)
    qml.CNOT(wires=[1, 3])


def build_circuit(
    ansatz: str,
    n_qubits: int = 4,
    n_layers: int = 2,
    noise_params: dict | None = None,
    backend: str = "default.qubit",
    diff_method: str = "backprop",
):
    """
    Build a PennyLane QNode with TorchLayer-compatible signature.

    Signature: circuit(inputs, weights)
      - inputs  : shape (n_qubits,), feature vector
      - weights : variational parameters per get_param_shape()

    Returns expectation values of PauliZ on ALL qubits.

    Parameters
    ----------
    ansatz : str
        One of ANSATZ_NAMES.
    n_qubits : int
        Number of qubits (default 4).
    n_layers : int
        Number of variational layers for SEL and Basic Entangler.
    noise_params : dict or None
        Dictionary of noise parameters (T1, T2, etc.). If present, 
        manual noise is applied layer-wise for strongly_entangling.
    backend : str
        "default.qubit" for ideal, "default.mixed" for noisy simulation.
    diff_method : str
        "backprop" for both backends.

    Returns
    -------
    qnode : callable
    """
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(inputs, weights):
        # Ry angle embedding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        if ansatz == "strongly_entangling":
            if noise_params:
                _apply_strongly_entangling_with_noise(weights, n_qubits, n_layers, noise_params)
            else:
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif ansatz == "basic_entangler":
            if noise_params:
                _apply_basic_entangler_with_noise(weights, n_qubits, n_layers, noise_params)
            else:
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        elif ansatz == "ttn":
            if noise_params:
                _apply_ttn_with_noise(weights, n_qubits, noise_params)
            else:
                _apply_ttn(weights, n_qubits)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        # Returns expectation values for only 2 qubits to match original paper architecture
        # and reduce catastrophic forgetting by forcing compact representations.
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    return circuit
