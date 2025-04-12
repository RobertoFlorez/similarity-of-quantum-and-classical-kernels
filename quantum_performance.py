from data_tools.get_dataset import *
from data_tools.tools import *
from circuits.circuits import *
from models.quantum_simulations import *
from pandas import DataFrame
from models.performance_simulations import get_performance_for_dic, get_performance_for_dic_wrapper
import time
import sys
import multiprocessing
import os

if os.name == 'posix':
    results_path = "/datax/results/"
elif os.name == 'nt':
    results_path = "./data/results/" # #server: /datax/results/

index = str(sys.argv[1])
index_experiment_list = int(sys.argv[2])
results_kernel_path = results_path + f"kernels_{index}_{index_experiment_list}.h5"
num_cores = int(sys.argv[3])

#time 
start = time.time()

#quantum_performance_relevant_columns = ['dataset_name', 'K_train', 'K_test', 'num_qubits', "encoding_circuit_name",
#                                        'num_layers', 'bandwidth', 'num_datapoints', 'method', 'executor_type',
#                                        'num_shots', 'gamma_original', "train_test_split_value"]

df = pd.DataFrame(read_experiment_dic_results(results_kernel_path, ignore_rho=True, ignore_Ks=False))
df_list_of_dicts = df.to_dict(orient='records')

results_performance_folder_path = results_path + f"SVC_performance_quantum_{index}_{index_experiment_list}"
if not os.path.exists(results_performance_folder_path):
    os.makedirs(results_performance_folder_path)

for idx, experiment in enumerate(df_list_of_dicts):
    results_performance_item_path = os.path.join(results_performance_folder_path, f"performance_quantum_{idx}")
    experiment["results_path"] = results_performance_item_path

print("Number of experiments:", len(df_list_of_dicts))
print("Number of cores:", num_cores)

#Constants
classical = False

include_svc = True
include_svr = False
include_krr = False

full_grid = "simple"
classification_problem = True


####################

if df_list_of_dicts[0]["dataset_name"] in regression_ds_list:
    include_svc = False
    include_svr = False
    include_krr = True

    full_grid = "simple"
    classification_problem = False

constants = (classical, include_svc, include_svr, include_krr, full_grid, classification_problem)
experiment_tuple_list = [(experiment_params, experiment_params["results_path"], constants) for experiment_params in df_list_of_dicts]



if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.starmap(get_performance_for_dic_wrapper, experiment_tuple_list)
        pool.close()
        pool.join()

        end = time.time()
        print("Performance calculations are done! Now merging the temporary files")
        #merge_temporary_files(results_performance_folder_path, 
        #                      results_path + f"performance_quantum_{index}_{index_experiment_list}.h5", ignore_errors=True)
        print("Everything done!")
        print("Time taken:", end - start)