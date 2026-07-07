"""
Multiclass classification model with quantum convolutional layers.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..ansatz.base import QCNNAnsatz

from ..layers import BatchedQuantumConv2D


class MultiClassQCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        encoding: str = "ry",
        ansatz: Optional[QCNNAnsatz] = None,
        measurement: str = "z",
        use_gpu: bool = False,
        readout_wires: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        qconv_kernel_size = 2
        qconv_stride = 1
        n_qubits = 4

        if input_size < 3:
            raise ValueError("input_size must be >= 3 for two k=2, s=1 convolutions")

        # Input image: (B, 3, N, N)
        # For MNIST (N=28): 28x28 -> 27x27 after this layer.
        self.downsample_conv = nn.Conv2d(3, 16, kernel_size=2, stride=1)

        self.downsample_nonlinearity = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Learnable channel bottleneck before the quantum layer.
        # We reduce channels so each quantum patch matches the encoding input
        # size exactly, avoiding fixed chunk-mean reduction in the qconv.
        if encoding == "dense":
            required_inputs = n_qubits * 3
        else:
            required_inputs = n_qubits
        patch_area = qconv_kernel_size * qconv_kernel_size
        if required_inputs % patch_area != 0:
            raise ValueError(
                "qconv input requirements are not divisible by patch area"
            )
        qconv_input_channels = required_inputs // patch_area
        self.qconv_channel_reduction = nn.Sequential(
            nn.Conv2d(16, qconv_input_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(qconv_input_channels),
        )

        # Quantum conv also uses k=2, s=1:
        # (N-1)x(N-1) -> (N-2)x(N-2). For MNIST: 27x27 -> 26x26.
        self.qconv = BatchedQuantumConv2D(
            kernel_size=qconv_kernel_size,
            stride=qconv_stride,
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
            use_gpu=use_gpu,
            readout_wires=readout_wires,
        )

        # Number of quantum output channels (one per measured wire).
        n_channels = self.qconv.n_channels

        # Normalize the quantum feature map: expectation values are bounded in
        # [-1, 1] and can drift in scale/mean across patches, so a BatchNorm
        # stabilizes the signal fed into pooling and the classical head.
        self.qconv_norm = nn.BatchNorm2d(n_channels)

        qconv_out_size = input_size - 2
        # Deterministic pooling target derived from input size.
        # For MNIST: qconv_out_size=26 -> pool_size=6.
        pool_size = max(1, qconv_out_size // 4)
        self.qconv_adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        min_width = max(2 * num_classes, 16)
        depth = 4
        # Each measured wire contributes a full pooled feature map.
        pooled_feature_dim = n_channels * pool_size * pool_size

        layer_sizes = [
            w for w in (pooled_feature_dim // (2 ** i) for i in range(1, depth + 1))
            if w >= min_width
        ]

        flat_layers: List[nn.Module] = [nn.Flatten()]

        loop_size = pooled_feature_dim
        for size in layer_sizes:
            flat_layers.append(nn.Linear(loop_size, size))
            flat_layers.append(nn.ReLU())
            flat_layers.append(nn.Dropout(0.2))
            loop_size = size

        flat_layers.append(nn.Linear(loop_size, num_classes))

        self.hidden_layers = nn.Sequential(*flat_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_conv(x)
        x = self.downsample_nonlinearity(x)
        x = self.qconv_channel_reduction(x)
        x = self.qconv(x)
        x = self.qconv_norm(x)
        x = self.qconv_adaptive_pool(x)
        x = self.hidden_layers(x)
        return x


class HybridQuantumMultiClassCNN(nn.Module):
    """
    Neural network with quantum convolutional kernels applied to image patches.
    Supports variable-sized images and different encoding strategies.
    Multiclass classification output.
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

        # 1. Classical Downsampling (process ALL pixels, output 16 channels)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # 2. Reduction to 1 channel for Quantum Layer
        self.rgb_reduction = nn.Conv2d(16, 1, kernel_size=1)

        if pool_size is None:
            # Recalculate input size after stride 4 downsampling
            if input_size is not None:
                feat_map_size = input_size // 4
                pool_size = (feat_map_size - kernel_size) // stride + 1
            else:
                pool_size = 8

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

        # Adaptive pooling to handle variable input sizes
        # Reduces to pool_size x pool_size regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # Classical layers for final processing
        # Input size depends on pool_size parameter
        layers: list[nn.Module] = [nn.Flatten()]
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

        layers.append(nn.Linear(input_dim, num_classes))

        self.classical = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)

        # Reduce 16 channels -> 1 channel (learnable)
        x = self.rgb_reduction(x)

        # Apply quantum convolution on the compressed features
        x = self.qconv(x)

        # Adaptive pooling to handle any size
        x = self.adaptive_pool(x)

        # Classical processing (outputs logits for each class)
        x = self.classical(x)

        return x
