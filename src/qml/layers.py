"""
Quantum convolutional layers for hybrid quantum-classical neural networks.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

from .ansatz.standard import StandardQCNNAnsatz
from .encoders import QuantumEncoder


class QuantumConv2D(nn.Module):
    """
    Quantum convolutional layer that applies QCNN as a sliding kernel over image
    patches.
    Similar to classical Conv2D but using quantum circuits.

    This is the reference sequential implementation. For performance,
    use BatchedQuantumConv2D.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        n_qubits=4,
        device_type="lightning.qubit",
        encoding="ry",
        ansatz=None,
        measurement="z",
    ):
        """
        Args:
            kernel_size: Size of the convolutional kernel
            stride: Stride for the convolution
            n_qubits: Number of qubits in the quantum circuit
            device_type: PennyLane device type
            encoding: Encoding strategy - 'rx', 'ry', 'rz', or 'dense'
            ansatz: QCNNAnsatz instance (defaults to StandardQCNNAnsatz)
            measurement: Measurement axis - 'x', 'y', or 'z' (default: 'z')
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.ansatz = (
            ansatz if ansatz is not None else StandardQCNNAnsatz(rotation_gate="ry")
        )

        # Validate and set measurement observable
        valid_measurements = ["x", "y", "z"]
        if measurement not in valid_measurements:
            raise ValueError(
                f"measurement must be one of {valid_measurements}, got '{measurement}'"
            )
        self.measurement = measurement
        self._observable_fn = {"x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}[
            measurement
        ]

        # Validate encoding option
        valid_encodings = ["rx", "ry", "rz", "dense"]
        if encoding not in valid_encodings:
            raise ValueError(
                f"encoding must be one of {valid_encodings}, got '{encoding}'"
            )

        # Try to use Lightning simulator for speed
        try:
            self.dev = qml.device(device_type, wires=n_qubits)
            print(
                f"Using {device_type} device with '{encoding}' encoding, "
                f"{type(self.ansatz).__name__}, measurement=Pauli{measurement.upper()}"
            )
        except Exception as e:
            print(
                f"Lightning device not available ({e}), falling back to default.qubit"
            )
            self.dev = qml.device("default.qubit", wires=n_qubits)
            print(
                f"Using '{encoding}' encoding, {type(self.ansatz).__name__}, "
                f"measurement=Pauli{measurement.upper()}"
            )

        # Quantum parameters based on ansatz requirements
        self.q_params = nn.Parameter(
            torch.randn(self.ansatz.n_layers, self.ansatz.n_params_per_layer) * 0.1
        )

    def encode_data(self, inputs):
        """
        Apply the selected encoding strategy to the input data.

        Args:
            inputs: Tensor of input values (length depends on encoding type)
        """
        if self.encoding == "rx":
            # X rotation encoding: one value per qubit
            for i in range(self.n_qubits):
                QuantumEncoder.rotation_x(inputs[i], wire=i)

        elif self.encoding == "ry":
            # Y rotation encoding: one value per qubit (default)
            for i in range(self.n_qubits):
                QuantumEncoder.rotation_y(inputs[i], wire=i)

        elif self.encoding == "rz":
            # Z rotation encoding: one value per qubit
            for i in range(self.n_qubits):
                QuantumEncoder.rotation_z(inputs[i], wire=i)

        elif self.encoding == "dense":
            # Dense encoding: 3 values per qubit
            # Requires 3 * n_qubits input values
            for i in range(self.n_qubits):
                values = inputs[i * 3 : (i + 1) * 3]
                QuantumEncoder.dense_encoding(values, wire=i)

    def qcnn_circuit(self, inputs, weights):
        """Quantum circuit for processing one patch."""
        # Encode patch into quantum state using selected encoding
        self.encode_data(inputs)

        # Apply the full QCNN ansatz (conv + pooling handled by the injected module)
        self.ansatz(weights)

        # Measurement on the final qubit using configured observable
        return qml.expval(self._observable_fn(self.n_qubits - 1))

    def extract_patches(self, x):
        """
        Extract patches from image tensor.

        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            patches: Tensor of shape
            (batch_size, n_patches_h, n_patches_w, kernel_size*kernel_size*channels)
        """
        batch_size, channels, height, width = x.shape

        # Calculate output dimensions
        out_h = (height - self.kernel_size) // self.stride + 1
        out_w = (width - self.kernel_size) // self.stride + 1

        patches = []
        for i in range(out_h):
            row_patches = []
            for j in range(out_w):
                # Extract patch
                h_start = i * self.stride
                w_start = j * self.stride
                patch = x[
                    :,
                    :,
                    h_start : h_start + self.kernel_size,
                    w_start : w_start + self.kernel_size,
                ]

                patch_flat = patch.flatten(start_dim=1)
                row_patches.append(patch_flat)
            patches.append(torch.stack(row_patches, dim=1))

        patches = torch.stack(patches, dim=1)
        return patches, out_h, out_w

    def forward(self, x):
        """
        Apply quantum kernel as a sliding window over the image.
        Sequential execution (slow).

        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, 1, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # Extract patches
        patches, out_h, out_w = self.extract_patches(x)
        # patches shape: (batch_size, out_h, out_w, patch_features)

        # Calculate required input size based on encoding
        if self.encoding == "dense":
            required_inputs = self.n_qubits * 3
        else:  # 'rx', 'ry', or 'rz'
            required_inputs = self.n_qubits

        # Process each patch through quantum circuit
        feature_maps = []
        for b in range(batch_size):
            batch_features = []
            for i in range(out_h):
                for j in range(out_w):
                    # Get patch
                    patch = patches[b, i, j]

                    # Reduce to required dimensions
                    if patch.shape[0] > required_inputs:
                        # Average pooling to reduce dimensions
                        patch_size = patch.shape[0] // required_inputs
                        patch_reduced = torch.stack(
                            [
                                patch[k * patch_size : (k + 1) * patch_size].mean()
                                for k in range(required_inputs)
                            ]
                        )
                    else:
                        # Pad if needed
                        patch_reduced = torch.cat(
                            [
                                patch,
                                torch.zeros(
                                    required_inputs - patch.shape[0],
                                    device=patch.device,
                                ),
                            ]
                        )[:required_inputs]

                    # Normalize to [-pi, pi] range
                    patch_norm = torch.tanh(patch_reduced) * np.pi

                    # Create QNode and execute
                    qnode = qml.QNode(
                        lambda weights: self.qcnn_circuit(patch_norm, weights),
                        self.dev,
                        interface="torch",
                    )
                    result = qnode(self.q_params)
                    batch_features.append(result.float())

            # Reshape to feature map
            feature_map = torch.stack(batch_features).reshape(out_h, out_w)
            feature_maps.append(feature_map)

        # Stack all batches: (batch_size, out_h, out_w) -> (batch_size, 1, out_h, out_w)
        output = torch.stack(feature_maps).unsqueeze(1)
        return output


class BatchedQuantumConv2D(QuantumConv2D):
    """
    Optimized Quantum Convolutional Layer using vectorized patch execution.
    Much faster than QuantumConv2D for large batches or images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the QNode for batched execution
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            self.encode_data(inputs)
            self.ansatz(weights)
            return qml.expval(self._observable_fn(self.n_qubits - 1))

        self.circuit_runner = circuit

    def forward(self, x):
        """
        Apply quantum kernel as a sliding window over the image.
        Batched execution for performance.

        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, 1, out_height, out_width)
        """
        batch_size, channels, height, width = x.shape

        # Extract patches
        patches, out_h, out_w = self.extract_patches(x)
        # patches shape: (batch_size, out_h, out_w, patch_features)

        # Flatten for batch processing: (Total_Patches, Features)
        total_patches = batch_size * out_h * out_w
        patches_flat = patches.view(total_patches, -1)

        # Calculate required input size based on encoding
        if self.encoding == "dense":
            required_inputs = self.n_qubits * 3
        else:  # 'rx', 'ry', or 'rz'
            required_inputs = self.n_qubits

        # Vectorized Pre-processing
        input_dim = patches_flat.shape[1]

        if input_dim > required_inputs:
            # Average pooling to reduce dimensions
            chunk_size = input_dim // required_inputs
            used_dim = required_inputs * chunk_size
            # Reshape to (Total, Required, Chunk) and mean over chunk
            inputs_reduced = (
                patches_flat[:, :used_dim]
                .view(total_patches, required_inputs, chunk_size)
                .mean(dim=2)
            )
        else:
            # Pad if needed
            padding = torch.zeros(
                total_patches, required_inputs - input_dim, device=x.device
            )
            inputs_reduced = torch.cat([patches_flat, padding], dim=1)

        # Normalize to [-pi, pi] range
        inputs_norm = torch.tanh(inputs_reduced) * np.pi

        # Transpose to (Features, Total_Patches) for PennyLane parameter broadcasting
        # PennyLane iterates over the first dimension of 'inputs' to map to wires/gates
        # so inputs[i] becomes the vector of feature i across all samples
        inputs_transposed = inputs_norm.t()

        # Execute Batched QNode
        # Returns shape: (Total_Patches,)
        results = self.circuit_runner(inputs_transposed, self.q_params)

        # Reshape to feature map: (batch_size, 1, out_h, out_w)
        output = results.view(batch_size, 1, out_h, out_w).float()

        return output


class BatchedGPUQuantumConv2D(BatchedQuantumConv2D):
    """
    Batched Quantum Conv2D specifically optimized for GPU execution.
    Uses default.qubit with backprop to enable Torch-native GPU simulation.
    Use this when lightning.gpu is not available but you want to run on GPU.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=2,
        n_qubits=4,
        encoding="ry",
        ansatz=None,
        measurement="z",
        **kwargs,
    ):
        # Force device_type to default.qubit which supports backprop on GPU
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            n_qubits=n_qubits,
            device_type="default.qubit",
            encoding=encoding,
            ansatz=ansatz,
            measurement=measurement,
            **kwargs,
        )

        # Re-define qnode with diff_method='backprop' to enable GPU support
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            self.encode_data(inputs)
            self.ansatz(weights)
            return qml.expval(self._observable_fn(self.n_qubits - 1))

        self.circuit_runner = circuit
