from data_tools.get_dataset import *
from data_tools.tools import *
from circuits.circuits import *
from models.quantum_simulations import *
from pandas import DataFrame
from experiment_inputs import experiment_list_total  # Import the configurations_list
import pandas as pd
from models.performance_simulations import get_performance_for_dic, get_performance_for_dic_wrapper, get_grid_list
from models.performance_simulations import get_params_combinations, get_performance_for_dic_classical_hyperparameters_wrapper
import sys
import time
import multiprocessing
import concurrent.futures
import os
import traceback
from threading import Thread
import functools


if os.name == 'posix':
    results_path = "/datax/results/"
elif os.name == 'nt':
    results_path = "./data/results/" # #server: /datax/results/
index = str(sys.argv[1])
index_experiment_list = int(sys.argv[2])
num_cores = int(sys.argv[3])

experiment_list = experiment_list_total[index_experiment_list]
#Constants
classical = True
include_svc = True
include_svr = False
include_krr = False
full_grid = "RBFandPolySV" 
#full_grid = "RBFSV" 
full_grid = "RBFandRBFPolySV"
classification_problem = True


####################



results_performance_folder_path = results_path + f"performance_classical_{index}_{index_experiment_list}"
if not os.path.exists(results_performance_folder_path):
    os.makedirs(results_performance_folder_path)

print("current path", os.getcwd())
#time 
start = time.time()

classical_performance_relevant_columns = ['dataset_name', 'num_qubits', 
                                        'bandwidth', 'num_datapoints', 'method', "train_test_split_value", "seed" ]
#df = pd.DataFrame(read_experiment_dic_results(results_kernel_path, ignore_rho=True), columns = classical_performance_relevant_columns)

df = pd.DataFrame(experiment_list, columns = classical_performance_relevant_columns)
df[["num_datapoints", "train_test_split_value", "seed"]] = pd.DataFrame(df["num_datapoints"].tolist(), index=df.index)

if experiment_list[0]["dataset_name"] in regression_ds_list:
    include_svc = False
    include_svr = False
    include_krr = True
    full_grid = "RBFandPolyKRR" 


    classification_problem = False
    print("check if benzene")

#df.sort_values(by=['bandwidth'], ascending=False, inplace=True, ignore_index=True) #Sort by bandwidth so that we can drop duplicates and keep the highest bandwidth (badnwidth = 1)
classical_relevant_subset = df.drop_duplicates(subset=['dataset_name', 'num_datapoints', 'num_qubits', "bandwidth", "seed"], ignore_index=True)


#only consider classical with bandwidth <= 10
#classical_relevant_subset = classical_relevant_subset[classical_relevant_subset["bandwidth"] <= 10]
#sort by bandwidth
classical_relevant_subset.sort_values(by=["num_qubits",  "bandwidth"], ascending=False, inplace=True, ignore_index=True)
#classical_relevant_subset.sort_values(by=['num_qubits'], ascending=False, inplace=True, ignore_index=True)

#create a new df which skips every 10th row
#classical_relevant_subset = classical_relevant_subset[::5]




classical_relevant_subset["C_list"], classical_relevant_subset["gamma_list"] = zip(*classical_relevant_subset.apply(lambda x: get_grid_list(full_grid, "", x["num_qubits"]), axis=1))
classical_relevant_subset["params_o"] = classical_relevant_subset.apply(lambda x: get_params_combinations(classical, x["C_list"], x["gamma_list"], full_grid, x["num_qubits"]), axis=1)
#params_o is a dictionary with keys as the index of the params and values as the params, explode it 
classical_relevant_subset_exploded = classical_relevant_subset.explode('params_o').reset_index(drop=True)
# Normalize the dictionaries in col1 into columns
classical_relevant_subset_exploded_normalized = pd.json_normalize(classical_relevant_subset_exploded['params_o'])
# Combine the normalized columns with the remaining columns
classical_relevant_subset = pd.concat([classical_relevant_subset_exploded.drop(columns=['params_o']), classical_relevant_subset_exploded_normalized], axis=1)
#create results_path for each experiment
classical_relevant_subset["results_path"] = classical_relevant_subset.apply(lambda x: os.path.join(results_performance_folder_path, f"performance_classical_{x.name}"), axis=1)
df_list_of_dicts = classical_relevant_subset.to_dict(orient='records')
constants = (classical, include_svc, include_svr, include_krr, full_grid, classification_problem)
####################


experiment_tuple_list = [(experiment_params, experiment_params["results_path"], constants) for experiment_params in df_list_of_dicts]

print("Number of experiments:", len(experiment_tuple_list))
print("Number of cores:", num_cores)


# def timeout_windows(timeout):
#     def deco(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
#             def newFunc():
#                 try:
#                     res[0] = func(*args, **kwargs)
#                 except Exception as e:
#                     res[0] = e
#             t = Thread(target=newFunc)
#             t.daemon = True
#             try:
#                 t.start()
#                 t.join(timeout)
#             except Exception as je:
#                 print ('error starting thread')
#                 raise je
#             ret = res[0]
#             if isinstance(ret, BaseException):
#                 raise ret
#             return ret
#         return wrapper
#     return deco

# print("Starting the experiment list in parallel")

# def get_performance_for_dic_windows_timeout(experiment_params, results_path, constants):
#     fun_timeout = timeout_windows(timeout=60*3)(get_performance_for_dic_classical_hyperparameters_wrapper)
#     try:
#         fun_timeout(experiment_params, results_path, constants)
#     except Exception:
#         print(f"Experiment {experiment_params} failed with error: {traceback.format_exc()}")
    


if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_cores) as pool:
        #pool.starmap(get_performance_for_dic_windows_timeout, experiment_tuple_list)
        pool.starmap(get_performance_for_dic_classical_hyperparameters_wrapper, experiment_tuple_list)

        pool.close()
        pool.join()

    print("Done!")
    end = time.time()
    print("Performance calculations are done! Now merging the temporary files")

    merge_temporary_files(results_performance_folder_path, results_performance_folder_path + "/classical_kernels.h5", clean_after_merge=False, ignore_errors=True)

    #merge_temporary_files(results_performance_folder_path, 
    #                      results_path_to_save + f"performance_classical_{index}_{index_experiment_list}.h5", ignore_errors=True)
    print("Everything done!")
    print("Time taken:", end - start)

"""

print("Starting the experiment list in parallel")
if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.starmap(get_performance_for_dic_wrapper, experiment_tuple_list)
        pool.close()
        pool.join()
        print("Done!")
        end = time.time()
        merge_temporary_files(results_performance_folder_path, results_performance_folder_path + "/classical_kernels.h5", clean_after_merge=True)
        print("Performance calculations are done! Now merging the temporary files")

        #merge_temporary_files(results_performance_folder_path, 
        #                      results_path_to_save + f"performance_classical_{index}_{index_experiment_list}.h5", ignore_errors=True)
        print("Everything done!")
        print("Time taken:", end - start)
"""