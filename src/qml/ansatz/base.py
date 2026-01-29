"""
Base class for QCNN ansatz implementations.
"""

from abc import ABC, abstractmethod


class QCNNAnsatz(ABC):
    """
    Abstract base class for QCNN ansatz with both convolution and pooling layers.
    Subclasses define the complete structure of a QCNN layer.
    """

    @abstractmethod
    def __call__(self, weights):
        """
        Apply the full QCNN ansatz (convolution + pooling layers).

        Args:
            weights: Full weights tensor for all layers
        """
        pass

    @property
    @abstractmethod
    def n_layers(self):
        """Total number of parametrized layers (ansatz blocks)."""
        pass

    @property
    @abstractmethod
    def n_params_per_layer(self):
        """Number of parameters per layer."""
        pass
