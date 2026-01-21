"""
Quantum encoding strategies for classical data.
"""
import pennylane as qml


class QuantumEncoder:
    """
    Modular quantum encoding strategies for classical data.
    """
    @staticmethod
    def rotation_x(value, wire):
        """
        Encode a single value using X rotation.
        
        Args:
            value: Single scalar value to encode
            wire: Qubit wire index
        """
        qml.RX(value, wires=wire)
    
    @staticmethod
    def rotation_y(value, wire):
        """
        Encode a single value using Y rotation.
        
        Args:
            value: Single scalar value to encode
            wire: Qubit wire index
        """
        qml.RY(value, wires=wire)
    
    @staticmethod
    def rotation_z(value, wire):
        """
        Encode a single value using Z rotation.
        
        Args:
            value: Single scalar value to encode
            wire: Qubit wire index
        """
        qml.RZ(value, wires=wire)
    
    @staticmethod
    def dense_encoding(values, wire):
        """
        Dense encoding: encode 3 values using X, Y, and Z rotations on one qubit.
        Each value is encoded in a different rotation axis.
        
        Args:
            values: Array/tensor of 3 values [x, y, z]
            wire: Qubit wire index
        """
        if len(values) < 3:
            raise ValueError("Dense encoding requires at least 3 values")
        qml.RX(values[0], wires=wire)
        qml.RY(values[1], wires=wire)
        qml.RZ(values[2], wires=wire)
