#from quantities_vs_num_dimensions_vs_bandwidth import *

from plotting.quantities_vs_num_dimensions_vs_bandwidth import *
import os 

simulation_set_index_kMNIST = [[(0, 13), (0, 15), (0, 16), (0, 28), (0, 33), (0, 5), (0, 7), (0, 8), (0, 27), (0, 31)], "kMNIST28"]


#simulation_sets = [ simulation_set_index_plasticc, simulation_set_index_kMNIST]

simulation_set_index_plasticc = [[(0, 1), (0, 3), (0, 4), (0, 25), (0, 30), (0, 9), (0, 11), (0, 12), (0, 26), (0, 32)], "plasticc"]

simulation_set_index_hiddenmanifold = [[(0, 55), (0, 56), (0, 57), (0, 58), (0, 59), (0, 60), (0, 61), (0, 62), (0, 63), (0, 64)], "hidden-manifold"]


simulation_sets = [ simulation_set_index_hiddenmanifold, simulation_set_index_plasticc, simulation_set_index_kMNIST]

names_of_metrics_long =  [kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str, expressibity_str]

for simulation_set in simulation_sets:
    dataset_name = simulation_set[1]

    kernel_to_compare = ["rbf_poly_2", "rbf"]

    path_to_save = f"./data/plots/{dataset_name}_numdimensions/"

    df_concat = pd.DataFrame()  
    for pair_index in simulation_set[0]:
        simulation_set_index = pair_index[0]
        dataset_set_index = pair_index[1]

        pd_geo_gramm_performance_both_ds1, pd_geo_top_results_enc_ds1, values_of_best_quantum_ds1, values_of_best_poly_ds1, N_train_ds1, dataset_name_str_ds1, encoding_circuit_name_str_ds1, kernel_type_to_keep_ds1, path_to_save_ds1 = load_and_return_preprocessed(simulation_set_index, dataset_set_index, dataset_name, kernel_to_compare)

        melted_ds_i, kernel_type_to_keep_du, num_qubit_order = top_enc_melting_preprocess(pd_geo_top_results_enc_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, kernel_to_compare, [kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str], use_kernel_acronym=True)

        num_qubit_order = [2, 6, 12, 16]

        # plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare[0], num_qubit_order, names_of_metrics_long, hide_ylabel=False, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)

        # plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare[0], num_qubit_order, names_of_metrics_long, hide_ylabel=True, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)


        # if len(kernel_to_compare) > 1:
        #     plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare[1], num_qubit_order, names_of_metrics_long, hide_ylabel=True, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)
        #     plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc_ds1, pd_geo_gramm_performance_both_ds1, kernel_type_to_keep_ds1, encoding_circuit_name_str_ds1, dataset_name_str_ds1, path_to_save_ds1, N_train_ds1, kernel_to_compare[1], num_qubit_order, names_of_metrics_long, hide_ylabel=True, show_plt = False, single_param_style="full", extra_number=1, only_classical=False)

        df_concat = pd.concat([df_concat, melted_ds_i])
        
    df_concat = df_concat.reset_index(drop=True)


    #create path to save if it does not exist
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    plot_metric_numdimension_normal(df_concat, "multi", dataset_name_str_ds1, path_to_save, N_train_ds1, kernel_to_compare, hide_xlabels=False)
    plot_metric_numdimension_normal(df_concat, "multi", dataset_name_str_ds1, path_to_save, N_train_ds1, kernel_to_compare, hide_xlabels=True)




