# %%
# %%
#set parent directory as package
import sys
#sys.path.append("../../")

from plotting.quantities_vs_num_dimensions_vs_bandwidth import *

simulation_set_index = sys.argv[1]
dataset_set_index = sys.argv[2]
classical_name = str(sys.argv[3])
kernel_to_compare = "rbf_poly_2"

hide = True

names_of_metrics_long =  [kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str, expressibity_str]

pd_geo_gramm_performance_both_ds1, pd_geo_top_results_enc_ds1, values_of_best_quantum_ds1, values_of_best_poly_ds1, N_train_ds1, dataset_name_str_ds1, encoding_circuit_name_str_ds1, kernel_type_to_keep_ds1, path_to_save_ds1 = load_and_return_preprocessed(simulation_set_index, dataset_set_index, classical_name, kernel_to_compare)

melted_ds1, kernel_type_to_keep_du, num_qubit_order = top_enc_melting_preprocess(pd_geo_top_results_enc_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, kernel_to_compare, [kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str])

num_qubit_order = [2, 6, 12, 16]

# plot_metric_numdimension(melted_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare, hide_xlabels=False)
# plot_metric_numdimension(melted_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare, hide_xlabels=True)

plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare, num_qubit_order, names_of_metrics_long, hide_ylabel=False, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)

kernel_to_compare = "rbf"
kernel_type_to_keep_ds1 = ['Quantum', kernel_to_compare]

melted_ds1, kernel_type_to_keep_du, num_qubit_order = top_enc_melting_preprocess(pd_geo_top_results_enc_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, kernel_to_compare, [kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str])

num_qubit_order = [2, 6, 12, 16]

plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare, num_qubit_order, names_of_metrics_long, hide_ylabel=False, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)



