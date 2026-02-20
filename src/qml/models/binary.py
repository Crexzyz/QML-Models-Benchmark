"""
Binary classification model with quantum convolutional layers.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..ansatz.base import QCNNAnsatz
from ..layers import BatchedGPUQuantumConv2D, BatchedQuantumConv2D, QuantumConv2D


class HybridQuantumCNN(nn.Module):
    """
    Neural network with quantum convolutional layers applied to image patches.
    Supports variable-sized images and different encoding strategies.
    Binary classification output.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 16,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            kernel_size: Size of quantum convolutional kernel
            stride: Stride for the quantum convolution
            pool_size: Size for adaptive pooling. If None and input_size is provided,
                      calculated automatically to preserve all features.
                      If both are None, defaults to 8.
            hidden_size: Number of neurons in the hidden layer(s) (default: 16).
                         Can be an int or a list of ints.
            encoding: Quantum encoding strategy - 'rx', 'ry', 'rz', or 'dense'
            ansatz: QCNNAnsatz instance (defaults to StandardQCNNAnsatz if None)
            measurement: Measurement axis - 'x', 'y', or 'z' (default: 'z')
            trainable_quantum: Whether to train quantum parameters (default: True)
            n_qubits: Number of qubits in quantum circuit (default: 4)
            input_size: Input image dimension (int). Used to calculate pool_size if
            not specified.
        """
        super().__init__()

        # Lightweight 1x1 convolution to learn optimal RGB mixing
        # Adds only 4 parameters but fixes the channel averaging data loss
        self.rgb_reduction = nn.Conv2d(3, 1, kernel_size=1)

        if pool_size is None:
            if input_size is not None:
                # Calculate output dimension of convolution to preserve all features
                # Output = (Input - Kernel) / Stride + 1
                pool_size = (input_size - kernel_size) // stride + 1
            else:
                pool_size = 8  # Fallback default

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
        layers: List[nn.Module] = [nn.Flatten()]
        input_dim = pool_size * pool_size

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = hidden_size

        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 1))

        self.classical = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Learnable RGB reduction to 1 channel (preserves info better than mean)
        x = self.rgb_reduction(x)

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
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 16,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
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
            input_size,
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
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 16,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
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
            input_size,
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
