"""
Hybrid quantum-classical neural network models.
"""

from .binary_classifier import HybridQuantumCNN
from .multiclass_classifier import HybridQuantumMultiClassCNN

__all__ = ["HybridQuantumCNN", "HybridQuantumMultiClassCNN"]
