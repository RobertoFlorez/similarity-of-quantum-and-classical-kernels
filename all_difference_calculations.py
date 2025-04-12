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
from models.geometric_diference_tools import calculate_and_write_kernel_difference_from_poly_results_SVC

if os.name == 'posix':
    results_path = "/datax/results/"
elif os.name == 'nt':
    results_path = "./data/results/" # #server: /datax/results/

kernel_index = sys.argv[1] #quantum kernel index 
dataset_index = sys.argv[2] # quantum kernel second index
classical_dataset_name = str(sys.argv[3])

short_load = False


#Quantum loading
performance_quantum_kernel_path = f"{results_path}SVC_performance_quantum_{kernel_index}_{dataset_index}/"
quantum_performance_pd = load_feather_folder_as_pd(performance_quantum_kernel_path)
results_file = f"{results_path}kernels_{kernel_index}_{dataset_index}.h5"
kernels_pd = pd.DataFrame(read_experiment_dic_results(results_file, ignore_rho = True, short_load = short_load, ignore_Ks=False))

#Classical loading
classical_performance_path = f"{results_path}{classical_dataset_name}_classical/classical_kernels.h5"
classical_performance_pd = pd.DataFrame(read_experiment_dic_results(classical_performance_path, short_load = short_load, ignore_Ks=False))
classical_performance_path_root = f"{results_path}performance_classical_{kernel_index}_{dataset_index}/"


# load_feather_folder_as_pd("../data/results/svc_performance_quantum_0_60")
#C_list = np.array([0.5, 32,64,128, 512, 1024*2, 1024*3])


geo_diff_without_C = False
C_list = []

if short_load:
    classical_performance_pd = classical_performance_pd[:100]


print("Starting gramm difference and geometrical difference together")
path_differences = performance_quantum_kernel_path + f"Vboth_difference_{kernel_index}_{dataset_index}"
calculate_and_write_kernel_difference_from_poly_results_SVC(kernels_pd.copy(), classical_performance_pd.copy(), C_list, path_differences, "both", quantum_performance_pd,  keep_all_bandwidths = True, geo_diff_without_C = geo_diff_without_C, skip_every_nth=1)


print("Starting geo difference top performing")
path_differences_top = performance_quantum_kernel_path + f"Vboth_difference_{kernel_index}_{dataset_index}_top_results"
calculate_and_write_kernel_difference_from_poly_results_SVC(kernels_pd.copy(), classical_performance_pd.copy(), C_list, path_differences_top, "both", quantum_performance_pd,  keep_all_bandwidths = False, geo_diff_without_C = geo_diff_without_C, skip_every_nth=1)

