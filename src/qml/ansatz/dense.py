"""
Dense QCNN ansatz implementation with multi-axis rotations.
"""

from typing import Literal
import pennylane as qml
from .base import QCNNAnsatz


class DenseQCNNAnsatz(QCNNAnsatz):
    """
    Abstract base class for Dense QCNN ansatz variants. Uses all three rotation
    axes (RX, RY, RZ) per qubit. Higher expressibility but more parameters.
    """

    def _apply_conv_block(self, params, wires) -> None:
        """Apply a dense two-qubit convolution block (6 params)."""
        qml.RX(params[0], wires=wires[0])
        qml.RX(params[3], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(params[1], wires=wires[0])
        qml.RY(params[4], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RZ(params[2], wires=wires[0])
        qml.RZ(params[5], wires=wires[1])

    @property
    def n_params_per_layer(self) -> Literal[6]:
        """Count of trainable classical parameters"""
        return 6


class DenseQCNNAnsatz4(DenseQCNNAnsatz):
    """
    Dense QCNN ansatz for 4 qubits.
    """

    def __call__(self, weights):
        """Apply the dense QCNN structure."""
        if len(weights) < self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} weight sets, got {len(weights)}"
            )

        # Layer 1: Full convolution
        self._apply_conv_block(weights[0], [0, 1])
        self._apply_conv_block(weights[1], [2, 3])
        self._apply_conv_block(weights[2], [0, 3])
        self._apply_conv_block(weights[3], [1, 2])

        # Pooling 1
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])

        # Layer 2
        self._apply_conv_block(weights[4], [1, 3])

        # Pooling 2
        qml.CNOT(wires=[1, 3])

    @property
    def n_layers(self) -> Literal[5]:
        return 5


class DenseQCNNAnsatz8(DenseQCNNAnsatz):
    """
    Dense QCNN ansatz for 8 qubits.
    """

    def __call__(self, weights):
        """Apply the dense QCNN structure."""
        if len(weights) < self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} weight sets, got {len(weights)}"
            )

        # Layer 1: Full convolution
        self._apply_conv_block(weights[0], [0, 1])
        self._apply_conv_block(weights[1], [2, 3])
        self._apply_conv_block(weights[2], [4, 5])
        self._apply_conv_block(weights[3], [6, 7])
        self._apply_conv_block(weights[4], [0, 7])
        self._apply_conv_block(weights[5], [1, 2])
        self._apply_conv_block(weights[6], [3, 4])
        self._apply_conv_block(weights[7], [5, 6])

        # Pooling 1
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[4, 5])
        qml.CNOT(wires=[6, 7])

        # Layer 2
        self._apply_conv_block(weights[8], [1, 3])
        self._apply_conv_block(weights[9], [5, 7])
        self._apply_conv_block(weights[10], [1, 7])
        self._apply_conv_block(weights[11], [3, 5])

        # Pooling 2
        qml.CNOT(wires=[1, 3])
        qml.CNOT(wires=[5, 7])

        # Layer 3
        self._apply_conv_block(weights[12], [3, 7])

        # Pooling 3
        qml.CNOT(wires=[3, 7])

    @property
    def n_layers(self) -> Literal[13]:
        return 13
