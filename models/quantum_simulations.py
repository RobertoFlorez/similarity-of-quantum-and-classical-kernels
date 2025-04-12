from data_tools.get_dataset import load_dataset
from data_tools.tools import  write_dic_results

from circuits.circuits import IQPLikeCircuit, Separable_rx, HubregtsenEncodingCircuit, YZ_CX_EncodingCircuit, ParamZFeatureMap
from models.manual_kernels import get_rho_matrices, K_FQK, K_PQK, variance_off_diagonal, separable_rx_gram_matrix
from models.spectral_bias_and_target_aligment import get_spectral_bias_tool

from squlearn.util import Executor
from qiskit.primitives import Estimator, Sampler
from squlearn.kernel.matrix import FidelityKernel, ProjectedQuantumKernel
from qiskit.quantum_info import  Statevector, DensityMatrix
from models.performance_simulations import LearnAndPredict_SVC, LearnAndPredict_SVR, LearnAndPredict_KRR
from sklearn.model_selection import ParameterGrid


import numpy as np




def get_KernelMatrix(dataset_name, num_datapoints, num_qubits, encoding_circuit_name, num_layers, bandwidth, method, executor_type, num_shots = np.NaN, hyperparameters = np.NaN, results_kernel_path = "../results/kernel_matrices.h5"):
    """
    dataset_name: str or np.array 
    num_qubits:
    encoding_circuit:
    num_layers:
    badnwidth:
    num_datapoints:
    method: "FQK" or "PQK" or "FQK_PQK"
    executor: "statevector" or "shots"
    """
    print(num_datapoints)

    metadata = {
        'dataset_name': dataset_name,
        'encoding_circuit_name': encoding_circuit_name,
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'bandwidth': bandwidth,
        'num_datapoints': num_datapoints[0],
        "train_test_split_value": num_datapoints[1], 
        "executor_type": executor_type,
    }
    if len(num_datapoints)>2:
        metadata["seed"] = num_datapoints[2]

    if executor_type == "shots":
            metadata["num_shots"] = num_shots
            rho_matrices = [np.nan, np.nan]

    if method == "FQK":
        metadata["method"] = "FQK"
    elif method == "PQK":
        metadata["method"] = "PQK"
        metadata["gamma_original"] = hyperparameters["gamma"]
    else:
        raise ValueError(f"Method {method} not supported yet")
    
    if dataset_name == "uniform" or dataset_name == "uniform_pi":
        X_train, X_test, y_train, y_test = load_dataset(dataset_name=dataset_name, num_qubits=num_qubits, num_datapoints=num_datapoints[0], train_test_split_value = num_datapoints[1], seed = metadata["seed"], optional_pca=False, higher_dim_original_distribution = False) #change to 0.8 because M.Schuld uses 0.8
    else:
        X_train, X_test, y_train, y_test = load_dataset(dataset_name=dataset_name, num_qubits=num_qubits, num_datapoints=num_datapoints[0], train_test_split_value = num_datapoints[1], seed = metadata["seed"]) #change to 0.8 because M.Schuld uses 0.8
    
    X_train, X_test, y_train, y_test = bandwidth* X_train, bandwidth* X_test,  y_train,  y_test


    if encoding_circuit_name == "IQPLikeCircuit":
        encoding_circuit = IQPLikeCircuit(num_qubits = num_qubits, num_layers=num_layers)
    elif encoding_circuit_name == "Separable_rx": #To be modified to implement analytical kernel instead
        encoding_circuit = Separable_rx(num_qubits, num_layers=num_layers)
    elif encoding_circuit_name == "HubregtsenEncodingCircuit":
        encoding_circuit = HubregtsenEncodingCircuit(num_qubits=num_qubits, num_features=num_qubits, num_layers=num_layers,)
        encoding_circuit.generate_initial_parameters(seed=1)
    elif encoding_circuit_name == "YZ_CX_EncodingCircuit":
        encoding_circuit = YZ_CX_EncodingCircuit(num_qubits=num_qubits, num_features=num_qubits, num_layers=num_layers)
        encoding_circuit.generate_initial_parameters(seed=1)
    elif encoding_circuit_name == "Z_Embedding":
        encoding_circuit = ParamZFeatureMap(num_qubits=num_qubits, num_features=num_qubits, num_layers=num_layers)
    else:
        raise ValueError(f"Encoding circuit {encoding_circuit_name} not supported yet, modify quantum_simulations.py to include it") 

    if encoding_circuit_name == "Separable_rx" and (executor_type == "statevector" or executor_type=="pennylane"):
        print("Calculating separable_rx_gram_matrix")
        K_train = separable_rx_gram_matrix(X_train, X_train, num_layers, method)
        K_test = separable_rx_gram_matrix(X_test, X_train, num_layers, method)

    else:
        if executor_type == "statevector" or "pennylane":
            if method == "FQK":
                if executor_type == "statevector":
                    executor = Executor("statevector")
                elif executor_type == "pennylane":
                    executor = Executor("pennylane")
                rho_matrices = [np.nan, np.nan] #I dont calculate the density matrices for FQK
                Kernel = FidelityKernel(encoding_circuit, executor=executor)
            elif method == "PQK":
                if executor_type == "statevector":
                    executor = Executor("statevector")
                elif executor_type == "pennylane":
                    executor = Executor("pennylane")            #Only for num_qubits < 8
                rho_matrices = [np.nan, np.nan] #I dont calculate the density matrices for FQK
                Kernel = ProjectedQuantumKernel(encoding_circuit, executor=executor, gamma=hyperparameters["gamma"])

            if encoding_circuit_name == "Z_Embedding":
                K_train = Kernel.evaluate_with_parameters(X_train, X_train, parameters= np.array([1 for _ in range(num_qubits*num_layers)]))
                K_test = Kernel.evaluate_with_parameters(X_test, X_train, parameters= np.array([1 for _ in range(num_qubits*num_layers)]))
            else:
                K_train = Kernel.evaluate(X_train, X_train)
                K_test = Kernel.evaluate(X_test, X_train)
        elif executor_type == "shots":
            print("shots probably return an error due to executor")
            rho_matrices = [np.nan, np.nan] #in the future, we could implement state tomography to get the density matrices
            if method == "FQK":
                executor = Executor(Sampler(),shots=num_shots, primitive_seed=1) # 
                Kernel = FidelityKernel(encoding_circuit, executor=executor)
                K_train = Kernel.evaluate(X_train, X_train)
                K_test = Kernel.evaluate(X_test, X_train)
            elif method == "PQK":
                executor = Executor(Estimator(), shots=num_shots, primitive_seed=1)
                Kernel = ProjectedQuantumKernel(encoding_circuit, executor=executor, gamma=hyperparameters["gamma"])
                K_train = Kernel.evaluate(X_train, X_train)
                K_test = Kernel.evaluate(X_test, X_train)


    if method == "FQK" or method == "PQK":

        varK_train = variance_off_diagonal(K_train)
        varK_test = np.var(K_test.flatten()) #varK_test includes the diagonal elements as well because, if not error can appear.
        metadata["varK_train"] = varK_train
        metadata["varK_test"] = varK_test

        #Calculating spectral-bias tools
        #print shapes 7
        
        eigenvalue_spectrum, eigenvectors, cumul, _ = get_spectral_bias_tool(K_train, X_train, y_train)
        metadata["eigenvalues"] = eigenvalue_spectrum
        metadata["eigenvectors"] = eigenvectors
        metadata["ck"] = cumul
        
        metadata["K_train"] = K_train
        metadata["K_test"] = K_test
        #metadata["X_train"] = X_train
        #metadata["X_test"] = X_test
        #metadata["y_train"] = y_train
        #metadata["y_test"] = y_test
        #metadata["density_matrices_train"] = rho_matrices[0]
        #metadata["density_matrices_test"] = rho_matrices[1]

        write_dic_results(results_kernel_path, metadata)


def get_KernelMatrix_dic_wrapper(dic):
    print(dic)
    return get_KernelMatrix(**dic)

