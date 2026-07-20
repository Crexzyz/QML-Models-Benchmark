"""
Quantum convolutional layers for hybrid quantum-classical neural networks.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

from .ansatz.standard import StandardQCNNAnsatz
from .encoders import QuantumEncoder


class BatchedQuantumConv2D(nn.Module):
    """
    Quantum convolutional layer using vectorized patch execution.
    Applies QCNN as a sliding kernel over image patches,
    similar to classical Conv2D but using quantum circuits.
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
        use_gpu=False,
        readout_wires=None,
    ):
        """
        Args:
            kernel_size: Size of the convolutional kernel
            stride: Stride for the convolution
            n_qubits: Number of qubits in the quantum circuit
            device_type: PennyLane device type (ignored if use_gpu=True)
            encoding: Encoding strategy - 'rx', 'ry', 'rz', or 'dense'
            ansatz: QCNNAnsatz instance (defaults to StandardQCNNAnsatz)
            measurement: Measurement axis - 'x', 'y', or 'z' (default: 'z')
            use_gpu: If True, use default.qubit with backprop for GPU support
            readout_wires: Wire(s) to measure, controlling the number of output
                channels. Accepts an int (single wire), an iterable of wire
                indices, or None. When None (default) only the last qubit
                (``n_qubits - 1``) is measured, preserving single-channel
                behaviour for pooling ansätze. Pass e.g. ``[0, 1, 2, 3]`` with a
                non-pooling ansatz to emit one channel per qubit.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.ansatz = (
            ansatz if ansatz is not None else StandardQCNNAnsatz(rotation_gate="ry")
        )
        self.use_gpu = use_gpu

        # Resolve and validate the readout wires (define output channel count).
        if readout_wires is None:
            readout_wires = [n_qubits - 1]
        elif isinstance(readout_wires, int):
            readout_wires = [readout_wires]
        else:
            readout_wires = list(readout_wires)
        if len(readout_wires) == 0:
            raise ValueError("readout_wires must select at least one wire")
        for w in readout_wires:
            if not (0 <= w < n_qubits):
                raise ValueError(
                    f"readout wire {w} out of range for {n_qubits} qubits"
                )
        self.readout_wires = readout_wires
        self.n_channels = len(readout_wires)

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

        # Override device if GPU mode is enabled
        if use_gpu:
            device_type = "default.qubit"

        # Try to use selected device (or fall back to default.qubit)
        try:
            self.dev = qml.device(device_type, wires=n_qubits)
            diff_method = "backprop" if use_gpu else None
            device_label = f"{device_type} (GPU mode)" if use_gpu else device_type
            print(
                f"Using {device_label} device with '{encoding}' encoding, "
                f"{type(self.ansatz).__name__}, measurement=Pauli{measurement.upper()}"
            )
        except Exception as e:
            print(
                f"Device '{device_type}' not available ({e}),"
                f" falling back to default.qubit"
            )
            self.dev = qml.device("default.qubit", wires=n_qubits)
            diff_method = "backprop" if use_gpu else None
            print(
                f"Using default.qubit with '{encoding}' encoding, "
                f"{type(self.ansatz).__name__}, "
                f"measurement=Pauli{measurement.upper()}"
            )

        # Quantum parameters based on ansatz requirements
        self.q_params = nn.Parameter(
            torch.randn(self.ansatz.n_layers, self.ansatz.n_params_per_layer) * 0.1
        )

        # Learnable input scale applied before the tanh squashing. Starting below
        # 1.0 keeps tanh in its near-linear region initially, avoiding saturated
        # (vanishing) gradients on the data-encoding rotations. The network can
        # grow this if sharper encoding is beneficial.
        self.input_scale = nn.Parameter(torch.tensor(0.5))

        # Define the QNode for batched execution
        qnode_kwargs = {"interface": "torch"}
        if diff_method:
            qnode_kwargs["diff_method"] = diff_method

        @qml.qnode(self.dev, **qnode_kwargs)
        def circuit(inputs, weights):
            self.encode_data(inputs)
            self.ansatz(weights)
            return [
                qml.expval(self._observable_fn(w)) for w in self.readout_wires
            ]

        self.circuit_runner = circuit
        self.reset_cost_metrics()

    def reset_cost_metrics(self):
        """Reset accumulated forward-pass cost metrics."""
        self._cost_metrics = {
            "forward_calls": 0,
            "circuit_executions": 0,
            "patches": 0,
            "layer_forward_time_s": 0.0,
            "circuit_time_s": 0.0,
        }

    def get_cost_metrics(self) -> dict:
        """Return accumulated forward-pass cost metrics."""
        return self._cost_metrics.copy()

    @staticmethod
    def _synchronize_if_cuda(tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

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

        # Vectorized patch extraction via unfold (im2col).
        # unfold -> (batch_size, channels * k * k, out_h * out_w)
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # -> (batch_size, out_h * out_w, channels * k * k)
        patches = patches.transpose(1, 2)
        # -> (batch_size, out_h, out_w, channels * k * k)
        patches = patches.reshape(batch_size, out_h, out_w, -1)
        return patches, out_h, out_w

    def forward(self, x):
        """
        Apply quantum kernel as a sliding window over the image.
        Batched execution for performance.

        Args:
            x: Tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, n_channels, out_height, out_width),
            where n_channels == len(readout_wires).
        """
        self._synchronize_if_cuda(x)
        layer_start = time.perf_counter()
        batch_size, channels, height, width = x.shape

        # Extract patches
        patches, out_h, out_w = self.extract_patches(x)
        # patches shape: (batch_size, out_h, out_w, patch_features)

        # Flatten for batch processing: (Total_Patches, Features)
        total_patches = batch_size * out_h * out_w
        patches_flat = patches.reshape(total_patches, -1)

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

        # Squash to a safe rotation range. A learnable pre-tanh scale keeps the
        # activation in its near-linear region at init (avoiding saturated,
        # vanishing gradients), and the amplitude is capped at pi/2 so that
        # distinct inputs do not alias under the 2*pi periodicity of the
        # encoding rotations (e.g. +pi and -pi collide for RY).
        inputs_norm = torch.tanh(inputs_reduced * self.input_scale) * (np.pi / 2)

        # Transpose to (Features, Total_Patches) for PennyLane parameter broadcasting
        # PennyLane iterates over the first dimension of 'inputs' to map to wires/gates
        # so inputs[i] becomes the vector of feature i across all samples
        inputs_transposed = inputs_norm.t()

        # Execute Batched QNode
        # Returns a list of length n_channels, each of shape (Total_Patches,).
        self._synchronize_if_cuda(x)
        circuit_start = time.perf_counter()
        results = self.circuit_runner(inputs_transposed, self.q_params)
        self._synchronize_if_cuda(x)
        circuit_time = time.perf_counter() - circuit_start

        # Stack the per-wire expectation values into a channel dimension:
        # (n_channels, Total_Patches)
        results = torch.stack(results, dim=0)

        # Reshape to feature map: (batch_size, n_channels, out_h, out_w)
        output = (
            results.reshape(self.n_channels, batch_size, out_h, out_w)
            .permute(1, 0, 2, 3)
            .float()
        )

        self._synchronize_if_cuda(x)
        layer_time = time.perf_counter() - layer_start
        self._cost_metrics["forward_calls"] += 1
        self._cost_metrics["circuit_executions"] += 1
        self._cost_metrics["patches"] += total_patches
        self._cost_metrics["layer_forward_time_s"] += layer_time
        self._cost_metrics["circuit_time_s"] += circuit_time
        return output
