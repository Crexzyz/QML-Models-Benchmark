"""
Multi-label classification model with quantum convolutional layers.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from qml.ansatz.base import QCNNAnsatz

from ..layers import BatchedQuantumConv2D


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
        use_gpu: bool = False,
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
            use_gpu: If True, use GPU-optimized quantum layer (default.qubit + backprop)
        """
        super().__init__()

        self.num_classes = num_classes

        # 1. Lightweight feature extractor (no pre-trained weights)
        # Conv2d(3->16, k=7, s=4, p=3): 640x640 -> 160x160
        # AvgPool2d(k=8, s=8):           160x160 -> 20x20
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=8),
        )

        # 2. Reduction: 16 channels -> 1 channel for Quantum Layer
        self.rgb_reduction = nn.Conv2d(16, 1, kernel_size=1)

        # 3. Fixed 4x4 pooling to keep FC layer size consistent
        fixed_pool_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((fixed_pool_dim, fixed_pool_dim))

        # Quantum convolutional layer (slides over image)
        self.qconv = BatchedQuantumConv2D(
            kernel_size=kernel_size,
            stride=stride,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
            use_gpu=use_gpu,
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
        # Feature extraction: 640x640 -> 20x20 (16 channels)
        x = self.backbone(x)

        # Reduce 16 channels -> 1 channel (learnable mix)
        x = self.rgb_reduction(x)

        # Apply quantum convolution (acts like Conv2D with quantum kernel)
        x = self.qconv(x)

        # Adaptive pooling to fixed 4x4 size
        x = self.adaptive_pool(x)

        # Classical processing (outputs logits for each class)
        x = self.classical(x)

        return x
