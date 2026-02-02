"""
Binary classification model with quantum convolutional layers.
"""

import torch.nn as nn

from ..layers import QuantumConv2D, BatchedQuantumConv2D, BatchedGPUQuantumConv2D


class HybridQuantumCNN(nn.Module):
    """
    Neural network with quantum convolutional kernels applied to image patches.
    Supports variable-sized images and different encoding strategies.
    Binary classification output.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        pool_size=8,
        hidden_size=16,
        encoding="ry",
        ansatz=None,
        measurement="z",
        trainable_quantum=True,
        n_qubits=4,
    ):
        """
        Args:
            kernel_size: Size of quantum convolutional kernel
            stride: Stride for the quantum convolution
            pool_size: Size for adaptive pooling
            hidden_size: Number of neurons in the hidden layer (default: 16)
            encoding: Quantum encoding strategy - 'rx', 'ry', 'rz', or 'dense'
            ansatz: QCNNAnsatz instance (defaults to StandardQCNNAnsatz if None)
            measurement: Measurement axis - 'x', 'y', or 'z' (default: 'z')
            trainable_quantum: Whether to train quantum parameters (default: True)
            n_qubits: Number of qubits in quantum circuit (default: 4)
        """
        super().__init__()

        # Quantum convolutional layer (slides over image)
        self.qconv = QuantumConv2D(
            kernel_size=kernel_size,
            stride=stride,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
        )

        # Control whether quantum parameters are trainable
        self.qconv.q_params.requires_grad = trainable_quantum

        # Adaptive pooling to handle variable input sizes
        # Reduces to pool_size x pool_size regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Classical layers for final processing
        # Input size depends on pool_size parameter
        self.classical = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pool_size * pool_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # Apply quantum convolution (acts like Conv2D with quantum kernel)
        x = self.qconv(x)

        # Adaptive pooling to handle any size
        x = self.adaptive_pool(x)

        # Classical processing
        x = self.classical(x)

        return x.reshape(-1)


class BatchedHybridQuantumCNN(HybridQuantumCNN):
    """
    Derived class using the optimized BatchedQuantumConv2D layer.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        pool_size=8,
        hidden_size=16,
        encoding="ry",
        ansatz=None,
        measurement="z",
        trainable_quantum=True,
        n_qubits=4,
    ):
        super().__init__(
            kernel_size,
            stride,
            pool_size,
            hidden_size,
            encoding,
            ansatz,
            measurement,
            trainable_quantum,
            n_qubits,
        )

        # Replace the qconv layer with the batched version
        self.qconv = BatchedQuantumConv2D(
            kernel_size=kernel_size,
            stride=stride,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
        )
        self.qconv.q_params.requires_grad = trainable_quantum


class BatchedGPUHybridQuantumCNN(HybridQuantumCNN):
    """
    Derived class using the optimized BatchedGPUQuantumConv2D layer.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        pool_size=8,
        hidden_size=16,
        encoding="ry",
        ansatz=None,
        measurement="z",
        trainable_quantum=True,
        n_qubits=4,
    ):
        super().__init__(
            kernel_size,
            stride,
            pool_size,
            hidden_size,
            encoding,
            ansatz,
            measurement,
            trainable_quantum,
            n_qubits,
        )

        # Replace the qconv layer with the batched GPU version
        self.qconv = BatchedGPUQuantumConv2D(
            kernel_size=kernel_size,
            stride=stride,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
        )
        self.qconv.q_params.requires_grad = trainable_quantum
