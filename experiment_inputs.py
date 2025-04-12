import numpy as np
import sys
from itertools import product


def get_experiment_combination_list(experimental_parameters):
    """
    Experimental parameters is a list of lists. Each list contains the possible values for a parameter.

    for example: 
    
    dataset_name_list = ["plasticc"]
    encoding_circuit_name_list = ["IQPLikeCircuit"] #"Separable_rx"
    num_qubits_list = [2,3,4,5]
    num_layers_list = [1, 3]
    bandwidth_list = [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1.0]
    num_datapoints_list = [800]
    method_list = ["FQK", "PQK"]
    executor_type_list = ["shots"]
    num_shots_list = [5000]
    gamma_list = [1.0]

    experimental_parameters = [dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list, ...]
    """
    experiment_list = []
    sorted_parameters = sorted(experimental_parameters, key=lambda x: x == "statevector")
    sorted_parameters = experimental_parameters
    for params in product(*sorted_parameters):
        dataset_name, encoding_circuit_name, num_qubits, num_layers, bandwidth, num_datapoints, method, executor_type, num_shots, gamma = params
        
        experiment = {
            "dataset_name": dataset_name,
            "encoding_circuit_name": encoding_circuit_name,
            "num_qubits": num_qubits,
            "num_layers": num_layers,
            "bandwidth": bandwidth,
            "num_datapoints": num_datapoints,
            "method": method,
            "executor_type": executor_type,
            "num_shots": num_shots,
            "hyperparameters": {
                "gamma": gamma
            }
        }
        experiment_list.append(experiment)
    return experiment_list



all_experiment_up_to_date = []



num_qubits_list = [2, 4, 6, 8, 10, 12, 14, 16] 
num_datapoints_list = [(400, 0.8, 1), (400, 0.8, 2), (400, 0.8, 3), (400, 0.8, 4), (400, 0.8, 5), (400, 0.8, 6)]
num_shots_list = [1]
gamma_list = [1.0]
executor_type_list = ["pennylane"]
bandwidth_list = np.logspace(-3, 1.5, 40).tolist()


quick_test = get_experiment_combination_list([["plasticc"], ["IQPLikeCircuit"], [2], [1],
                           [0.25],  [(400, 0.8, 6)], ["FQK"], executor_type_list,
                            num_shots_list, gamma_list])


#Experiment L=2 FQK plasticc

num_layers_list = [2]
method_list = ["FQK"]
dataset_name_list = ["plasticc"]

encoding_circuit_name_list = ["IQPLikeCircuit"]
plasticc_IQP_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HamEvol"]
plasticc_HamEvol_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
plasticc_YZ_CX_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
plasticc_Hubregtsen_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
plasticc_Z_embedding_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

#Experiment L=2 FQK kmnist28
dataset_name_list = ["kMNIST28"]


encoding_circuit_name_list = ["IQPLikeCircuit"]
kMNIST_IQP_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HamEvol"]
kMNIST_HamEvol_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
kMNIST_YZ_CX_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
kMNIST_Hubregtsen_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
kMNIST_Z_embedding_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

method_list = ["PQK"]
dataset_name_list = ["plasticc"]

encoding_circuit_name_list = ["IQPLikeCircuit"]
plasticc_IQP_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HamEvol"]
plasticc_HamEvol_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
plasticc_YZ_CX_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
plasticc_Hubregtsen_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
plasticc_Z_embedding_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

#Experiment L=2 FQK kmnist28
dataset_name_list = ["kMNIST28"]


encoding_circuit_name_list = ["IQPLikeCircuit"]
kMNIST_IQP_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HamEvol"]
kMNIST_HamEvol_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
kMNIST_YZ_CX_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
kMNIST_Hubregtsen_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
kMNIST_Z_embedding_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])


num_qubits_list = [2, 4, 6, 8, 10, 12, 14, 16] 
num_datapoints_list = [(400, 0.8, 1), (400, 0.8, 2), (400, 0.8, 3), (400, 0.8, 4), (400, 0.8, 5), (400, 0.8, 6)]
num_shots_list = [1]
gamma_list = [1.0]
executor_type_list = ["pennylane"]
bandwidth_list = np.logspace(-3, 1.5, 40).tolist()

num_layers_list = [4]
method_list = ["FQK"]
dataset_name_list = ["plasticc"]

encoding_circuit_name_list = ["IQPLikeCircuit"]
plasticc_IQP_4L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

plasticc_IQP_6L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, [6],
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

plasticc_IQP_multiL_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, [4], [1,2,4,6,8, 10, 12, 14, 16],
                            bandwidth_list, [(800, 0.9, 1)], method_list, executor_type_list,
                            num_shots_list, gamma_list])

uniform_sep_rx_multiL_FQK = get_experiment_combination_list([["uniform_pi"], ["Separable_rx"], [4], [1,2,4,6,8, 10, 12, 14, 16],
                            np.logspace(-3, 1.5, 40).tolist(), [(800, 0.9, 1)], ["FQK"], executor_type_list,
                            num_shots_list, gamma_list])
#uniform_sep_rx_multiL_FQK = get_experiment_combination_list([["uniform_pi"], ["Separable_rx"], [2, 4, 6, 8, 10, 12, 14, 16] , [4],


dataset_name_list = ["kMNIST28"]


num_qubits_list = [2, 4, 6, 8, 10, 12, 14, 16] 
num_datapoints_list = [(400, 0.8, 1), (400, 0.8, 2), (400, 0.8, 3), (400, 0.8, 4), (400, 0.8, 5), (400, 0.8, 6)]
num_shots_list = [1]
gamma_list = [1.0]
executor_type_list = ["pennylane"]
bandwidth_list = np.logspace(-3, 1.5, 40).tolist()



encoding_circuit_name_list = ["IQPLikeCircuit"]
kMNIST_IQP_4L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

num_layers_list = [6]

encoding_circuit_name_list = ["IQPLikeCircuit"]
kMNIST_IQP_6L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])
num_layers_list = [1]

encoding_circuit_name_list = ["IQPLikeCircuit"]
plasticc_IQP_1L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

dataset_name_list = ["kMNIST28"]


encoding_circuit_name_list = ["IQPLikeCircuit"]
kMNIST_IQP_6L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])


toy_datapoint_list =  [(800, 0.9, 1), (800, 0.9, 2), (800, 0.9, 3)]
separable_rx_2L_uniform_many_points = get_experiment_combination_list([["uniform_pi"], ["Separable_rx"], [2, 4, 6, 8, 10, 12, 14, 16]  , [1],
                            np.logspace(-3, 1.5, 40).tolist(), toy_datapoint_list, ["FQK"], executor_type_list,
                            num_shots_list, gamma_list])


toy_datapoint_list =  [(800, 0.9, 1)] #102 is with 3500
analytical_uniform_seprx = get_experiment_combination_list([["uniform_pi"], ["Separable_rx"], [1, 2, 3, 4, 5, 6]  , [1],
                            np.logspace(-3, 1.5, 40).tolist(), toy_datapoint_list, ["FQK"], executor_type_list,
                            num_shots_list, gamma_list])


num_qubits_list = [2, 4, 6, 8, 10, 12, 14, 16] 
num_datapoints_list = [(400, 0.8, 1), (400, 0.8, 2), (400, 0.8, 3), (400, 0.8, 4), (400, 0.8, 5), (400, 0.8, 6)]
num_shots_list = [1]
gamma_list = [1.0]
executor_type_list = ["pennylane"]
bandwidth_list = np.logspace(-3, 1.5, 40).tolist()

num_layers_list = [2]
method_list = ["FQK"]
dataset_name_list = ["plasticc"]

encoding_circuit_name_list = ["Separable_rx"]
plasticc_sepRX_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])
plasticc_sepRX_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, ["PQK"], executor_type_list,
                            num_shots_list, gamma_list])
kMNIST_sepRX_2L_FQK = get_experiment_combination_list([["kMNIST28"], encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])
kMNIST_sepRX_2L_PQK = get_experiment_combination_list([["kMNIST28"], encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, ["PQK"], executor_type_list,
                            num_shots_list, gamma_list])


num_datapoints_list = [(150, 0.75, 1), (150, 0.75, 2)]

###################################33 


########### hidden-manifold


num_qubits_list = [2, 4, 6, 8, 10, 12, 14, 16] 
num_datapoints_list = [(400, 0.8, 1), (400, 0.8, 2), (400, 0.8, 3), (400, 0.8, 4), (400, 0.8, 5), (400, 0.8, 6)]
num_shots_list = [1]
gamma_list = [1.0]
executor_type_list = ["pennylane"]
bandwidth_list = np.logspace(-3, 1.5, 40).tolist()

#Experiment L=2 FQK kmnist28
dataset_name_list = ["pennylane_hidden-manifold"]


encoding_circuit_name_list = ["IQPLikeCircuit"]
hiddenmanifold_IQP_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
hiddenmanifold_YZ_CX_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
hiddenmanifold_Hubregtsen_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
hiddenmanifold_Z_embedding_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Separable_rx"]
hiddenmanifold_sepRX_2L_FQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])


method_list = ["PQK"]
dataset_name_list = ["pennylane_hidden-manifold"]

encoding_circuit_name_list = ["IQPLikeCircuit"]
hiddenmanifold_IQP_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["YZ_CX_EncodingCircuit"]
hiddenmanifold_YZ_CX_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["HubregtsenEncodingCircuit"]
hiddenmanifold_Hubregtsen_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])

encoding_circuit_name_list = ["Z_Embedding"]
hiddenmanifold_Z_embedding_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, method_list, executor_type_list,
                            num_shots_list, gamma_list])
encoding_circuit_name_list = ["Separable_rx"]
hiddenmanifold_sepRX_2L_PQK = get_experiment_combination_list([dataset_name_list, encoding_circuit_name_list, num_qubits_list, num_layers_list,
                            bandwidth_list, num_datapoints_list, ["PQK"], executor_type_list,
                            num_shots_list, gamma_list])



uniform_IQP_multiL_FQK = get_experiment_combination_list([["uniform_pi"], ["IQPLikeCircuit"], [4], [1,2,4,6,8, 10, 12, 14, 16], bandwidth_list, [(800, 0.9, 1)], ["FQK"], executor_type_list, num_shots_list, gamma_list])
plasticc_sepRX_multiL_FQK = get_experiment_combination_list([["plasticc"], ["Separable_rx"], [4], [1,2,4,6,8, 10, 12, 14, 16], bandwidth_list, [(800, 0.9, 1)], ["FQK"], executor_type_list, num_shots_list, gamma_list])

uniform_Hubreg_multiL_FQK = get_experiment_combination_list([["uniform_pi"], ["HubregtsenEncodingCircuit"], [4], [1,2,4,6,8, 10, 12, 14, 16], bandwidth_list, [(800, 0.9, 1)], ["FQK"], executor_type_list, num_shots_list, gamma_list])

plasticc_Hubreg_multiL_FQK = get_experiment_combination_list([["plasticc"], ["HubregtsenEncodingCircuit"], [4], [1,2,4,6,8, 10, 12, 14, 16], bandwidth_list, [(800, 0.9, 1)], ["FQK"], executor_type_list, num_shots_list, gamma_list])


experiment_list_total = [
                         quick_test, # 0 
                         plasticc_IQP_2L_FQK, # 1
                         plasticc_HamEvol_2L_FQK,  # 2
                         plasticc_YZ_CX_2L_FQK, # 3
                         plasticc_Hubregtsen_2L_FQK, # 4
                         kMNIST_IQP_2L_FQK, # 5
                         kMNIST_HamEvol_2L_FQK, # 6
                         kMNIST_YZ_CX_2L_FQK, # 7
                         kMNIST_Hubregtsen_2L_FQK, # 8
                         plasticc_IQP_2L_PQK, # 9
                         plasticc_HamEvol_2L_PQK,  # 10
                         plasticc_YZ_CX_2L_PQK, # 11
                         plasticc_Hubregtsen_2L_PQK, # 12
                         kMNIST_IQP_2L_PQK, # 13
                         kMNIST_HamEvol_2L_PQK, # 14
                         kMNIST_YZ_CX_2L_PQK, # 15
                         kMNIST_Hubregtsen_2L_PQK, # 16
                         separable_rx_2L_uniform_many_points, # 17
                         "", # 18
                         analytical_uniform_seprx, # 19
                         "", # 20 delte ----------------------------------------
                         plasticc_IQP_4L_FQK, # 21
                         kMNIST_IQP_4L_FQK, # 22
                         plasticc_IQP_6L_FQK, # 23
                         kMNIST_IQP_6L_FQK, # 24
                         plasticc_sepRX_2L_FQK, # 25
                         plasticc_sepRX_2L_PQK, # 26
                         kMNIST_sepRX_2L_FQK, # 27
                         kMNIST_sepRX_2L_PQK, # 28
                         "", # 29
                         plasticc_Z_embedding_2L_FQK, # 30
                         kMNIST_Z_embedding_2L_FQK, # 31
                         plasticc_Z_embedding_2L_PQK, # 32
                         kMNIST_Z_embedding_2L_PQK, # 33
                         "", # 34
                         "", # 35
                         "", # 36
                         "", # 37
                         "", # 38
                         "", # 39
                         "", # 40
                         "", # 41
                         "", # 42
                         "", # 43
                         "", # 44
                         "", # 45
                         "", # 46
                         "", # 47
                         "", # 48
                         "", # 49
                         "", # 50
                         "", # 51
                         "", # 52
                         "", # 53
                         "", # 54
                         hiddenmanifold_IQP_2L_FQK, # 55
                         hiddenmanifold_YZ_CX_2L_FQK, # 56
                         hiddenmanifold_Hubregtsen_2L_FQK, # 57
                         hiddenmanifold_Z_embedding_2L_FQK, # 58
                         hiddenmanifold_sepRX_2L_FQK, # 59
                         hiddenmanifold_IQP_2L_PQK, # 60
                         hiddenmanifold_YZ_CX_2L_PQK, # 61
                         hiddenmanifold_Hubregtsen_2L_PQK, # 62
                         hiddenmanifold_Z_embedding_2L_PQK, # 63
                         hiddenmanifold_sepRX_2L_PQK, # 64              
                         plasticc_IQP_1L_FQK, # 65     
                         plasticc_IQP_multiL_FQK, # 66      
                         uniform_sep_rx_multiL_FQK, # 67
                         uniform_IQP_multiL_FQK, # 68
                         plasticc_sepRX_multiL_FQK, # 69
                         uniform_Hubreg_multiL_FQK, # 70
                         plasticc_Hubreg_multiL_FQK, # 71
                         ]



