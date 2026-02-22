"""
Multiclass classification model with quantum convolutional layers.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..ansatz.base import QCNNAnsatz
from ..layers import BatchedGPUQuantumConv2D, BatchedQuantumConv2D, QuantumConv2D


class HybridQuantumMultiLabelCNN(nn.Module):
    """
    Neural network with quantum convolutional kernels applied to image patches.
    Supports variable-sized images and different encoding strategies.
    Multi-label classification output.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 64,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            num_classes: Number of output classes
            kernel_size: Size of quantum convolutional kernel
            stride: Stride for the quantum convolution
            pool_size: Size for adaptive pooling. If None and input_size is provided,
                      calculated automatically to preserve all features.
            hidden_size: Number of neurons in the hidden layer(s) (default: 64).
                         Can be an int or a list of ints.
            encoding: Quantum encoding strategy - 'rx', 'ry', 'rz', or 'dense'
            ansatz: QCNNAnsatz instance (defaults to StandardQCNNAnsatz if None)
            measurement: Measurement axis - 'x', 'y', or 'z' (default: 'z')
            trainable_quantum: Whether to train quantum parameters (default: True)
            n_qubits: Number of qubits in quantum circuit (default: 4)
            input_size: Input image dimension (int). Used to calculate pool_size
            if not specified.
        """
        super().__init__()

        self.num_classes = num_classes

        # 1. Use Pre-trained ResNet18 as Feature Extractor (Transfer Learning)
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final classification layer (fc) and average pooling layer
        # Keep layers up to layer4 to get spatial features (512 channels)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze backbone to avoid destroying pre-trained features during initial
        # training
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Reduction: 512 ResNet channels -> 1 channel for Quantum Layer
        self.rgb_reduction = nn.Conv2d(512, 1, kernel_size=1)

        # 3. Fixed size pooling since ResNet output is consistent
        # For ResNet18:
        # Input 640x640 -> 20x20
        # Input 256x256 -> 8x8
        # We pool to 4x4 regardless of input size to keep FC layer consistent
        # and act as spatial pyramid pooling
        fixed_pool_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((fixed_pool_dim, fixed_pool_dim))

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

        # Classical layers for final processing
        # Input size depends on pool_size parameter
        layers: list[nn.Module] = [nn.Flatten()]
        input_dim = fixed_pool_dim * fixed_pool_dim

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = hidden_size

        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.classical = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-process: 640x640 -> 20x20 semantic map (with 512 channels)
        # Using pre-trained ResNet backbone
        x = self.backbone(x)

        # Reduce 512 channels -> 1 channel (learnable mix)
        x = self.rgb_reduction(x)

        # Apply quantum convolution (acts like Conv2D with quantum kernel)
        x = self.qconv(x)

        # Adaptive pooling to fixed 4x4 size
        x = self.adaptive_pool(x)

        # Classical processing (outputs logits for each class)
        x = self.classical(x)

        return x


class BatchedHybridQuantumMultiLabelCNN(HybridQuantumMultiLabelCNN):
    """
    Derived class using the optimized BatchedQuantumConv2D layer for multi-label.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 64,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
    ):
        super().__init__(
            num_classes,
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


class BatchedGPUHybridQuantumMultiLabelCNN(HybridQuantumMultiLabelCNN):
    """
    Derived class using the optimized BatchedGPUQuantumConv2D layer for multi-label.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 2,
        stride: int = 2,
        pool_size: Optional[int] = None,
        hidden_size: Union[int, List[int]] = 64,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        trainable_quantum: bool = True,
        n_qubits: int = 4,
        input_size: Optional[int] = None,
    ):
        super().__init__(
            num_classes,
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
