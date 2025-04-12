from squlearn.encoding_circuit import *

import numpy as np
#import reduce

from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
#import zzfeaturemap






def IQPLikeCircuit(num_qubits, num_layers, qiskit_way = False):
    """
    IQPLikeCircuit(num_qubits, num_layers)
    Returns a circuit that is similar to the one used in IQP.
    """
    from qiskit.circuit.library import ZZFeatureMap
    from functools import reduce

    def self_product(x: np.ndarray) -> float:
        """
        Define a function map from R^n to R.

        Args:
            x: data

        Returns:
            float: the mapped value
        """
        #product of all elements in x
        return reduce(lambda a, b: a * b, x)
    if qiskit_way:
        return  ZZFeatureMap(num_qubits, reps = num_layers, data_map_func = self_product)
    return  QiskitEncodingCircuit(ZZFeatureMap, feature_dimension=num_qubits, reps = num_layers, data_map_func = self_product)



def Separable_rx(num_qubits, num_layers):
    """
    Separable_rx(num_qubits, num_layers)
    Returns a circuit that is similar to the one used in IQP.
    """
    fmap = LayeredEncodingCircuit(num_qubits=num_qubits, num_features=num_qubits)
    for layer in range(num_layers):
        fmap.Rx("x")
    return fmap
