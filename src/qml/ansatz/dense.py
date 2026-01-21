"""
Dense QCNN ansatz implementation with multi-axis rotations.
"""
import pennylane as qml
from .base import QCNNAnsatz


class DenseQCNNAnsatz(QCNNAnsatz):
    """
    Dense QCNN ansatz using all three rotation axes (RX, RY, RZ) per qubit.
    Higher expressibility but more parameters.
    """
    
    def _apply_conv_block(self, params, wires):
        """Apply a dense two-qubit convolution block (6 params)."""
        # First qubit: RX, RY, RZ
        qml.RX(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[0])
        qml.RZ(params[2], wires=wires[0])
        # Second qubit: RX, RY, RZ
        qml.RX(params[3], wires=wires[1])
        qml.RY(params[4], wires=wires[1])
        qml.RZ(params[5], wires=wires[1])
        # Entanglement
        qml.CNOT(wires=[wires[0], wires[1]])
    
    def __call__(self, weights):
        """Apply the dense QCNN structure."""
        if len(weights) < self.n_layers:
            raise ValueError(f"Expected {self.n_layers} weight sets, got {len(weights)}")
        
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
    def n_layers(self):
        return 5
    
    @property
    def n_params_per_layer(self):
        return 6
