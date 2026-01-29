"""
Standard QCNN ansatz implementation with single-axis rotations.
"""

import pennylane as qml
from .base import QCNNAnsatz


class StandardQCNNAnsatz(QCNNAnsatz):
    """
    Standard 4-qubit QCNN ansatz with RY rotations.

    Structure:
        Layer 1 (Conv): Ansatz on [0,1], [2,3], [0,3], [1,2]
        Pooling 1: CNOT [0→1], CNOT [2→3]
        Layer 2 (Conv): Ansatz on [1,3]
        Pooling 2: CNOT [1→3]
        Measurement: PauliZ on qubit 3
    """

    def __init__(self, rotation_gate="ry"):
        """
        Args:
            rotation_gate: Type of rotation gate - 'rx', 'ry', or 'rz'
        """
        valid_gates = ["rx", "ry", "rz"]
        if rotation_gate not in valid_gates:
            raise ValueError(
                f"rotation_gate must be one of {valid_gates}, got '{rotation_gate}'"
            )
        self.rotation_gate = rotation_gate
        self._gate_fn = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}[rotation_gate]

    def _apply_conv_block(self, params, wires):
        """Apply a two-qubit convolution block."""
        self._gate_fn(params[0], wires=wires[0])
        self._gate_fn(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        self._gate_fn(params[2], wires=wires[0])
        self._gate_fn(params[3], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        self._gate_fn(params[4], wires=wires[0])
        self._gate_fn(params[5], wires=wires[1])

    def __call__(self, weights):
        """Apply the full QCNN structure."""
        # Validate weights shape
        if len(weights) < self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} weight sets, got {len(weights)}"
            )

        # Layer 1: Convolution on all qubit pairs
        self._apply_conv_block(weights[0], [0, 1])
        self._apply_conv_block(weights[1], [2, 3])
        self._apply_conv_block(weights[2], [0, 3])
        self._apply_conv_block(weights[3], [1, 2])

        # Pooling 1: Reduce 4 qubits to 2 active qubits
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])

        # Layer 2: Convolution on remaining active qubits
        self._apply_conv_block(weights[4], [1, 3])

        # Pooling 2: Final reduction
        qml.CNOT(wires=[1, 3])

    @property
    def n_layers(self):
        return 5

    @property
    def n_params_per_layer(self):
        return 6
