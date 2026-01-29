"""
Hybrid quantum-classical neural network models.
"""

from .binary import HybridQuantumCNN
from .multiclass import HybridQuantumMultiClassCNN

__all__ = ["HybridQuantumCNN", "HybridQuantumMultiClassCNN"]
