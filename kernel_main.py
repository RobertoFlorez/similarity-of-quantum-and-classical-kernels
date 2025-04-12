
from data_tools.get_dataset import *
from data_tools.tools import *
from circuits.circuits import *
from models.quantum_simulations import *
from experiment_inputs import experiment_list_total  # Import the configurations_list
import sys
import multiprocessing 
import os
import time


start = time.time()

if os.name == 'posix':
    results_path = "/datax/results/"
elif os.name == 'nt':
    results_path = "./data/results/" # #server: /datax/results/index = str(sys.argv[1])

index = str(sys.argv[1])
index_experiment_list = int(sys.argv[2])
num_cores = int(sys.argv[3])

experiment_list = experiment_list_total[index_experiment_list]
results_kernel_folder_path = results_path + f"kernels_{index}_{index_experiment_list}"
if not os.path.exists(results_kernel_folder_path):
    os.makedirs(results_kernel_folder_path)

for idx, experiment in enumerate(experiment_list):
    results_kernel_item_path = os.path.join(results_kernel_folder_path, f"kernels_{idx}.h5")
    experiment["results_kernel_path"] = results_kernel_item_path

print("Number of experiments:", len(experiment_list))
print("Index of experiment list:", index_experiment_list)
print("Number of cores:", num_cores)

#save experiment_list to a file as npy
np.save(results_path + f"experiment_list_{index}_{index_experiment_list}.npy", experiment_list)

print("Starting the experiment list in parallel")
if __name__ == "__main__":
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(get_KernelMatrix_dic_wrapper, experiment_list)
        pool.close()
        pool.join()
        print("Kernel calculations are done! Now merging the temporary files")
        merge_temporary_files(results_kernel_folder_path, results_path + f"kernels_{index}_{index_experiment_list}.h5")
        remove_temporary_files(results_kernel_folder_path)
        end = time.time()
        print("Everything done!")
        print("Time taken:", end - start)

