import sys
import os
import torch
import time
import pennylane as qml
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from qml.layers import QuantumConv2D, BatchedQuantumConv2D, BatchedGPUQuantumConv2D
from qml.ansatz.standard import StandardQCNNAnsatz

def compare_implementations():
    print("--------------------------------------------------")
    print("Comparing Sequential vs Batched (CPU) vs Batched (GPU) QuantumConv2D")
    print("--------------------------------------------------")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device for GPU model: {device}")
    
    # Setup
    batch_size = 16  # Increased load
    channels = 1 
    height = 28
    width = 28
    n_qubits = 4
    
    # Create random input
    # Fix seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, channels, height, width)
    # GPU Version needs input on device
    x_gpu = x.clone().to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Initialize Layers
    ansatz = StandardQCNNAnsatz()
    
    print("Initializing layers...")
    
    # 1. Sequential (Slow)
    layer_sequential = QuantumConv2D(
        n_qubits=n_qubits,
        ansatz=ansatz,
        stride=1,
        encoding='ry'
    )
    
    # 2. Batched CPU (Fast)
    layer_batched_cpu = BatchedQuantumConv2D(
        n_qubits=n_qubits,
        ansatz=ansatz,
        stride=1,
        encoding='ry'
    )
    
    # 3. Batched GPU (Fastest?)
    layer_batched_gpu = BatchedGPUQuantumConv2D(
        n_qubits=n_qubits,
        ansatz=ansatz,
        stride=1,
        encoding='ry'
    ).to(device)
    
    # FORCE WEIGHTS TO BE IDENTICAL
    with torch.no_grad():
        layer_batched_cpu.q_params.data = layer_sequential.q_params.data.clone()
        layer_batched_gpu.q_params.data = layer_sequential.q_params.data.to(device)
    
    print("Weights synchronized.")
    
    # ------------------------------------------------
    # Test Sequential (Sequential)
    # ------------------------------------------------
    print("\nRunning Sequential (Slow) Version (CPU)...")
    start_time = time.time()
    with torch.no_grad():
        out_sequential = layer_sequential(x)
    end_time = time.time()
    time_sequential = end_time - start_time
    print(f"Sequential Time: {time_sequential:.4f} seconds")
    
    # ------------------------------------------------
    # Test Batched CPU
    # ------------------------------------------------
    print("\nRunning Batched Version (CPU)...")
    start_time = time.time()
    with torch.no_grad():
        out_batched_cpu = layer_batched_cpu(x)
    end_time = time.time()
    time_batched_cpu = end_time - start_time
    print(f"Batched CPU Time: {time_batched_cpu:.4f} seconds")
    
    # ------------------------------------------------
    # Test Batched GPU
    # ------------------------------------------------
    print(f"\nRunning Batched Version ({device})...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out_batched_gpu = layer_batched_gpu(x_gpu)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    time_batched_gpu = end_time - start_time
    print(f"Batched GPU Time: {time_batched_gpu:.4f} seconds")
    
    # ------------------------------------------------
    # Comparisons
    # ------------------------------------------------
    speedup_batched = time_sequential / time_batched_cpu
    speedup_gpu = time_sequential / time_batched_gpu
    speedup_batched_gpu = time_batched_cpu / time_batched_gpu
    
    print(f"\nSpeedup Batched vs Sequential: {speedup_batched:.2f}x")
    print(f"Speedup GPU vs Sequential:     {speedup_gpu:.2f}x")
    print(f"Speedup GPU vs Batched CPU:    {speedup_batched_gpu:.2f}x")
    
    # Check correctness
    diff_batched = torch.abs(out_sequential - out_batched_cpu).max().item()
    diff_gpu = torch.abs(out_sequential - out_batched_gpu.cpu()).max().item()
    
    print(f"\nMax Diff Batched: {diff_batched}")
    print(f"Max Diff GPU:     {diff_gpu}")
    
    if diff_batched < 1e-4 and diff_gpu < 1e-4:
        print(">> VALIDATION PASSED: All results are identical.")
    else:
        print(">> VALIDATION FAILED: Results are different!")

if __name__ == "__main__":
    compare_implementations()