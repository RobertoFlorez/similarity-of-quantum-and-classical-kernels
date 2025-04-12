import pandas as pd
import numpy as np
from data_tools.tools import read_experiment_dic_results, write_dic_results, load_feather_folder_as_pd, rename_kernel


#######
#Obtains the eigenvalues of the Gramm matrix for the quantum and classical kernels and saves them in a specific pickle file, optimized to work with the notebook "plotting/eigenvalues_plots.ipynb"

#First, run from the WinterKernel root folder:
#py -3 -m plotting.create_gramm_comparison_file_best_performance_direct_values
#then use notebook to plot

#######

def path_names(simulation_set_index, dataset_set_index):
    path_all = f"./data/results/svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}.feather"
    path_optimal = f"./data/results/svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}_top_results.feather"
    return path_all, path_optimal




#path classical with boths
paths_differences_all = [path_names(0, 1)[0], path_names(0, 3)[0], path_names(0, 4)[0], path_names(0, 9)[0], path_names(0, 11)[0], path_names(0, 12)[0], path_names(0, 5)[0], path_names(0, 7)[0], path_names(0, 8)[0], path_names(0, 13)[0], path_names(0, 15)[0], path_names(0, 16)[0], path_names(0, 25)[0], path_names(0, 26)[0], path_names(0, 27)[0], path_names(0, 28)[0], path_names(0, 30)[0], path_names(0, 31)[0], path_names(0, 32)[0], path_names(0, 33)[0], path_names(0, 55)[0], path_names(0, 56)[0], path_names(0, 57)[0], path_names(0, 58)[0], path_names(0, 59)[0], path_names(0, 60)[0], path_names(0, 61)[0], path_names(0, 62)[0], path_names(0, 63)[0], path_names(0, 64)[0]]
paths_differences_optimal = [path_names(0, 1)[1], path_names(0, 3)[1], path_names(0, 4)[1], path_names(0, 9)[1], path_names(0, 11)[1], path_names(0, 12)[1], path_names(0, 5)[1], path_names(0, 7)[1], path_names(0, 8)[1], path_names(0, 13)[1], path_names(0, 15)[1], path_names(0, 16)[1], path_names(0, 25)[1], path_names(0, 26)[1], path_names(0, 27)[1], path_names(0, 28)[1], path_names(0, 30)[1], path_names(0, 31)[1], path_names(0, 32)[1], path_names(0, 33)[1], path_names(0, 55)[1], path_names(0, 56)[1], path_names(0, 57)[1], path_names(0, 58)[1], path_names(0, 59)[1], path_names(0, 60)[1], path_names(0, 61)[1], path_names(0, 62)[1], path_names(0, 63)[1], path_names(0, 64)[1] ]

# paths_differences_all = [path_names(0, 1)[0], path_names(0, 4)[0], path_names(0, 9)[0], path_names(0, 12)[0], path_names(0, 5)[0], path_names(0, 8)[0], path_names(0, 13)[0], path_names(0, 16)[0], path_names(0, 25)[0], path_names(0, 26)[0], path_names(0, 27)[0], path_names(0, 28)[0], path_names(0, 30)[0], path_names(0, 31)[0], path_names(0, 32)[0], path_names(0, 33)[0]]
# paths_differences_optimal = [path_names(0, 1)[1], path_names(0, 4)[1], path_names(0, 9)[1], path_names(0, 12)[1], path_names(0, 5)[1], path_names(0, 8)[1], path_names(0, 13)[1], path_names(0, 16)[1], path_names(0, 25)[1], path_names(0, 26)[1], path_names(0, 27)[1], path_names(0, 28)[1], path_names(0, 30)[1], path_names(0, 31)[1], path_names(0, 32)[1], path_names(0, 33)[1]]

c_value_index_list = [27] #13

num_qubits_to_keep = [2, 4, 8, 10, 16]
c_values_to_keep = [1]


c_value_list = np.logspace(-3, 1.5, 40).tolist()

#c_values_to_keep.append(c_value_list[0])
#c_values_to_keep.append(c_value_list[38])



df_global = pd.DataFrame()
for i in range(len(paths_differences_all)):
    df_temp_all = pd.read_feather(paths_differences_all[i])
    
    #including c=1 kernels
    df_temp_for_c_high_filtered = df_temp_all[df_temp_all["bandwidth"].isin(c_values_to_keep)]
    df_temp_for_c_high_filtered = df_temp_for_c_high_filtered[df_temp_for_c_high_filtered["num_qubits"].isin(num_qubits_to_keep)]

    df_temp_for_c_high_filtered["bandwidth_poly"] = df_temp_for_c_high_filtered["bandwidth"]
    df_temp_for_c_high_filtered["bandwidth_quantum"] = df_temp_for_c_high_filtered["bandwidth"]
    df_temp_for_c_high_filtered["classical_kernel_name"] = df_temp_for_c_high_filtered.apply(lambda x: rename_kernel(x, False), axis=1)
    df_temp_for_c_high_filtered["quantum_kernel_name"] = df_temp_for_c_high_filtered.apply(lambda x: rename_kernel(x, True), axis=1)
    
    df_temp_for_c_high_filtered["optimal c"] = df_temp_for_c_high_filtered.apply(lambda x: False, axis=1)
    df_global = pd.concat([df_global, df_temp_for_c_high_filtered], ignore_index = True, axis=0)


    #including top performance 
    df_temp_optimal = pd.read_feather(paths_differences_optimal[i])
    df_temp_for_optimal_c_filtered = df_temp_optimal[df_temp_optimal["num_qubits"].isin(num_qubits_to_keep)]

    df_temp_for_optimal_c_filtered["classical_kernel_name"] = df_temp_for_optimal_c_filtered.apply(lambda x: rename_kernel(x, False), axis=1)
    df_temp_for_optimal_c_filtered["quantum_kernel_name"] = df_temp_for_optimal_c_filtered.apply(lambda x: rename_kernel(x, True), axis=1)
    df_temp_for_optimal_c_filtered["optimal c"] = df_temp_for_optimal_c_filtered.apply(lambda x: True, axis=1)

    
    df_global = pd.concat([df_global, df_temp_for_optimal_c_filtered], ignore_index = True, axis=0)


name = "df_global_to_compare_gramm_matrices.pkl"
df_global.to_pickle(f"./data/results/{name}")


#save df_global to

#pd.DataFrame(read_experiment_dic_results("../data/results/kernels_0_59.h5", short_load=True, ignore_Ks=False))

#save df_global with pickle


