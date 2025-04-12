import pandas as pd
import numpy as np
from data_tools.tools import read_experiment_dic_results, write_dic_results, load_feather_folder_as_pd, rename_kernel


#######
#Obtains the eigenvalues of the Gramm matrix for the quantum and classical kernels and saves them in a specific pickle file, optimized to work with the notebook "plotting/analytical_plots_Figs4_A10_A11.ipynb"

#First, run from the WinterKernel root folder:
#py -3 -m plotting.create_eigenvalue_comparison_file
#then use notebook to plot

#######



def path_names(simulation_set_index, dataset_set_index):
    path_all = f"./data/results/svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}.feather"
    path_optimal = f"./data/results/svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}_top_results.feather"
    return path_all, path_optimal


paths_differences_all = [path_names(0, 1)[0], path_names(0, 3)[0], path_names(0, 4)[0], path_names(0, 9)[0], path_names(0, 11)[0], path_names(0, 12)[0], path_names(0, 5)[0], path_names(0, 7)[0], path_names(0, 8)[0], path_names(0, 13)[0], path_names(0, 15)[0], path_names(0, 16)[0], path_names(0, 25)[0], path_names(0, 26)[0], path_names(0, 27)[0], path_names(0, 28)[0], path_names(0, 30)[0], path_names(0, 31)[0], path_names(0, 32)[0], path_names(0, 33)[0], path_names(0, 55)[0], path_names(0, 56)[0], path_names(0, 57)[0], path_names(0, 58)[0], path_names(0, 59)[0], path_names(0, 60)[0], path_names(0, 61)[0], path_names(0, 62)[0], path_names(0, 63)[0], path_names(0, 64)[0]]
paths_differences_optimal = [path_names(0, 1)[1], path_names(0, 3)[1], path_names(0, 4)[1], path_names(0, 9)[1], path_names(0, 11)[1], path_names(0, 12)[1], path_names(0, 5)[1], path_names(0, 7)[1], path_names(0, 8)[1], path_names(0, 13)[1], path_names(0, 15)[1], path_names(0, 16)[1], path_names(0, 25)[1], path_names(0, 26)[1], path_names(0, 27)[1], path_names(0, 28)[1], path_names(0, 30)[1], path_names(0, 31)[1], path_names(0, 32)[1], path_names(0, 33)[1], path_names(0, 55)[1], path_names(0, 56)[1], path_names(0, 57)[1], path_names(0, 58)[1], path_names(0, 59)[1], path_names(0, 60)[1], path_names(0, 61)[1], path_names(0, 62)[1], path_names(0, 63)[1], path_names(0, 64)[1] ]


paths_classical = ["./data/results/kMNIST28_classical/classical_kernels.h5", "./data/results/plasticc_classical/classical_kernels.h5", "./data/results/hidden-manifold_classical/classical_kernels.h5"]



num_qubits_to_keep = [2, 4, 6, 8, 10, 14, 16]


c_value_list = np.logspace(-3, 1.5, 40).tolist()

c_values_to_keep = [1] #c=1
c_values_to_keep.append(c_value_list[0]) #very small c
c_values_to_keep.append(c_value_list[38]) #very large c



df_global = pd.DataFrame()

for i in range(len(paths_differences_all)):
    df_temp_all = pd.read_feather(paths_differences_all[i])
    df_temp_all["bandwidth_poly"] = df_temp_all["bandwidth"]
    df_temp_all["bandwidth_quantum"] = df_temp_all["bandwidth"]
    
    #including c=1 kernels
    df_temp_for_c_high_filtered = df_temp_all[df_temp_all["bandwidth"].isin(c_values_to_keep)]
    df_temp_for_c_high_filtered = df_temp_for_c_high_filtered[df_temp_for_c_high_filtered["num_qubits"].isin(num_qubits_to_keep)]

    df_temp_for_c_high_filtered["classical_kernel_name"] = df_temp_for_c_high_filtered.apply(lambda x: rename_kernel(x, False), axis=1)
    df_temp_for_c_high_filtered["quantum_kernel_name"] = df_temp_for_c_high_filtered.apply(lambda x: rename_kernel(x, True), axis=1)
    
    df_temp_for_c_high_filtered["optimal c_quantum"] = df_temp_for_c_high_filtered.apply(lambda x: False, axis=1)
    df_global = pd.concat([df_global, df_temp_for_c_high_filtered], ignore_index = True, axis=0)


    #including top performance 
    df_temp_optimal = pd.read_feather(paths_differences_optimal[i])
    
    df_temp_for_optimal_c_filtered = df_temp_optimal[df_temp_optimal["num_qubits"].isin(num_qubits_to_keep)]

    df_temp_for_optimal_c_filtered["classical_kernel_name"] = df_temp_for_optimal_c_filtered.apply(lambda x: rename_kernel(x, False), axis=1)
    df_temp_for_optimal_c_filtered["quantum_kernel_name"] = df_temp_for_optimal_c_filtered.apply(lambda x: rename_kernel(x, True), axis=1)
    df_temp_for_optimal_c_filtered["optimal c_quantum"] = df_temp_for_optimal_c_filtered.apply(lambda x: True, axis=1)

    
    df_global = pd.concat([df_global, df_temp_for_optimal_c_filtered], ignore_index = True, axis=0)

    #remove bandwidth_column 
    df_global = df_global.drop(columns=["bandwidth"])


#load classical kernels
df_global_classical = pd.DataFrame()
for i in range(len(paths_classical)):
    df_temp_all = pd.DataFrame(read_experiment_dic_results(paths_classical[i], ignore_rho = True, ignore_Ks = True, short_load = False))
    df_temp_all["classical_kernel_name"] = df_temp_all.apply(lambda x: rename_kernel(x, False), axis=1)
    df_temp_all = df_temp_all[df_temp_all["num_qubits"].isin(num_qubits_to_keep)]
    df_temp_all["bandwidth_poly"] = df_temp_all["bandwidth"]

    print(df_temp_all.keys())
    #including fixed c kernels
    df_temp_for_c_high_filtered = df_temp_all[df_temp_all["bandwidth_poly"].isin(c_values_to_keep)]
    
    df_temp_for_c_high_filtered = df_temp_for_c_high_filtered[["bandwidth_poly", "num_qubits", "classical_kernel_name", "seed", "dataset_name", "eigenvalues", "ck"]]
    df_temp_for_c_high_filtered["optimal c_classical"] = df_temp_for_c_high_filtered.apply(lambda x: False, axis=1)


    df_global_classical = pd.concat([df_global_classical, df_temp_for_c_high_filtered], ignore_index = True, axis=0)

    #including top performance
    df_temp_optimal = df_temp_all.sort_values(by=['roc_auc_score'], ascending=False, ignore_index=True).copy()
    df_temp_for_optimal_c_filtered = df_temp_optimal.drop_duplicates(subset=["dataset_name", "seed", "num_qubits", "classical_kernel_name"])
    df_temp_for_optimal_c_filtered = df_temp_for_optimal_c_filtered[["bandwidth_poly",  "num_qubits", "classical_kernel_name", "seed", "dataset_name", "eigenvalues", "ck"]]
    df_temp_for_optimal_c_filtered["optimal c_classical"] = df_temp_for_optimal_c_filtered.apply(lambda x: True, axis=1)
    df_global_classical = pd.concat([df_global_classical, df_temp_for_optimal_c_filtered], ignore_index = True, axis=0)


#merge df_global and df_global_classical

df_global = pd.merge(df_global, df_global_classical, how='inner', 
                     on=['num_qubits', 'bandwidth_poly', 'classical_kernel_name', "seed", "dataset_name"], suffixes=('_quantum', '_classical'))

df_global["bandwidth_classical"] = df_global["bandwidth_poly"]

name = "df_global_to_compare_gramm_matrices_bandwidth_eigen.pkl"

df_global.to_pickle(f"./data/results/{name}")

print(df_global.head())

#save df_global to

#pd.DataFrame(read_experiment_dic_results("../data/results/kernels_0_59.h5", short_load=True, ignore_Ks=False))

#save df_global with pickle


