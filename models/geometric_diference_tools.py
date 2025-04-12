import numpy as np
from scipy.linalg import sqrtm # matrix square root
from data_tools.get_dataset import load_dataset, regression_ds_list
from data_tools.tools import rename_kernel
import pandas as pd




def geometric_difference(
        K1: np.ndarray,
        K2: np.ndarray,
        lambda_reg: float = 0.0,
        ) -> tuple[float,float]:

        """ Function for calculating the geometric difference 
        (from Power of data in quantum machine learning paper)
        between two Kernel matrices K1 and K2
    
        g_12 = g_gen = sqrt(inf_norm(sqrt(K1)sqrt(K2)(K2+lambda_reg)^-2 sqrt(K2)sqrt(K1))  F19
        g_tra = lambda_reg*sqrt(inf_norm(sqrt(K1)(K2+lambda_reg)^-2 sqrt(K1)))

        typically g12 = g(K1,K2) = g(K1 = KC, K2 = KQ) 

        In the power of data: 
        g_gen  is reported  given in Equation (F19) with the largest λ such
        that the training error g_tra ≤ 0.045.

        lambda_reg should be chosen such that g_tra is small

        Args:
            K1 (np.ndarray): Kernel matrix 1
            K2 (np.ndarray): Kernel matrix 2
            lambda_reg (float): Regularization parameter
        """

        from scipy.linalg import sqrtm

        try:
            sqrt_K1 = sqrtm(K1)
            sqrt_K2 = sqrtm(K2)
            if lambda_reg == 0.0:
                K2_shift_inv = np.linalg.inv(K2)
            else:
                K2_shift_inv = np.linalg.inv(K2 + np.eye(K2.shape[0]) * lambda_reg) #MSc Thesis 
            K2_shift_inv_squared = K2_shift_inv @ K2_shift_inv  
            g12_gen = np.sqrt(np.linalg.norm(sqrt_K1 @ sqrt_K2 @ K2_shift_inv_squared @ sqrt_K2 @ sqrt_K1, ord=2))   
            if lambda_reg == 0.0:
                g12_tra = 0
            else:
                g12_tra = lambda_reg * np.sqrt(np.linalg.norm(sqrt_K1 @ K2_shift_inv_squared @ sqrt_K1, ord=2))
        except Exception as e:
            print(e)
            g12_gen = np.nan
            g12_tra = np.nan

        return np.real_if_close(g12_gen), np.real_if_close(g12_tra)


def kernel_difference_metric(K_poly, K_quantum):
    """
    Function for calculating the difference between two kernel matrices
    using the Frobenius norm and the distance metric d_G defined by maria schuld
    """
    try:
        d_G = np.sum((K_poly - K_quantum)**2)/K_poly.shape[0]**2
        d_G_mod = np.sum((K_poly - K_quantum)**2)/np.linalg.norm(K_quantum, "fro")**2

        frobenius_difference = np.linalg.norm(K_poly - K_quantum, "fro")/np.linalg.norm(K_quantum, "fro")
        #normalized_difference = np.linalg.norm(K_poly - K_quantum, ord=2)
    except:
        frobenius_difference = np.nan
        d_G = np.nan
        d_G_mod = np.nan
    return frobenius_difference, d_G, d_G_mod




def K_fro_norms(K_poly, K_quantum):
    """
    Function for calculating the difference between two kernel matrices
    using the Frobenius norm and the distance metric d_G defined by maria schuld
    """
    try:
        K_poly_norm = np.linalg.norm(K_poly, "fro")
        K_quantum_norm = np.linalg.norm(K_quantum, "fro")
        #normalized_difference = np.linalg.norm(K_poly - K_quantum, ord=2)
    except:
        K_poly_norm = np.nan
        K_quantum_norm = np.nan
    return K_poly_norm, K_quantum_norm



def kernel_difference_from_poly_results_SVC(df_kernel, df_poly, C_list, df_performance_quantum=None, difference_option="both", keep_all_bandwidths = True, geo_diff_without_C = False, skip_every_nth=1):
    """
    Function for calculating the gram matrix difference between the kernel matrices of df_kernel and df_poly over the regularization parameters C_list, and returning the results in a pandas dataframe.
    If df_performance_quantum is provided, it calculates only for the top-performing entries.

    Suggestion to load the dataframes:
    df_kernel = pd.DataFrame(
        read_experiment_dic_results(kernel_file, ignore_Ks=False))[["K_test", "K_train", "bandwidth", "eigenvalues", "ck", "dataset_name", "encoding_circuit_name", "method", "num_layers", "num_qubits"]]
    df_poly = pd.DataFrame(read_experiment_dic_results(results_file + "merged.h5", ignore_Ks=False))
    """
    import swifter
    
    


    df_kernel = df_kernel[["K_test", "K_train", "bandwidth", "eigenvalues", "ck", "dataset_name", "encoding_circuit_name", "method", "num_layers", "num_qubits", "seed"]]

    if df_kernel["dataset_name"].iloc[0] in regression_ds_list:
        classification_problem = False
    else:
        classification_problem = True

    if df_performance_quantum is not None:

        if classification_problem:
            try:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "degree", "gamma", "kernel", "num_qubits", "seed", "roc_auc_score", "K_max_eig", "C"]]
            except:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "gamma", "kernel", "num_qubits", "seed", "roc_auc_score", "K_max_eig", "C"]]
            df_performance_quantum = df_performance_quantum[["bandwidth", "eigenvalues", "ck", "dataset_name", "encoding_circuit_name", "method", "num_layers", "num_qubits", "seed", "roc_auc_score", "K_max_eig", "C"]]
        else:
            try:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "degree", "gamma", "kernel", "num_qubits", "seed", "mse", "K_max_eig", "alpha"]]
            except:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "gamma", "kernel", "num_qubits", "seed", "mse", "K_max_eig", "alpha"]]
            df_performance_quantum = df_performance_quantum[["bandwidth", "eigenvalues", "ck", "dataset_name", "encoding_circuit_name", "method", "num_layers", "num_qubits", "seed", "mse", "K_max_eig", "alpha"]]
        
        quantum_performance_with_kernels = pd.merge(
            df_performance_quantum,
            df_kernel,
            on=["num_qubits", "bandwidth", "seed", "dataset_name", "encoding_circuit_name", "method", "num_layers"],
            suffixes=('', '_extra')
        )


        columns_to_drop = [col for col in quantum_performance_with_kernels.columns if col.endswith('_extra')]
        quantum_performance_with_kernels = quantum_performance_with_kernels.drop(columns=columns_to_drop)

    
        if classification_problem:
            quantum_performance_with_kernels = quantum_performance_with_kernels.sort_values(by=["roc_auc_score", "bandwidth"], ascending=False)
            df_poly = df_poly.sort_values(by=["roc_auc_score", "bandwidth"], ascending=False)
        else:
            quantum_performance_with_kernels = quantum_performance_with_kernels.sort_values(by=["mse", "bandwidth"], ascending=[True, False])
            df_poly = df_poly.sort_values(by=["mse", "bandwidth"], ascending=[True, False])
    
        if keep_all_bandwidths == False:
            quantum_performance_with_kernels = quantum_performance_with_kernels.drop_duplicates(subset=["num_qubits", "seed", "dataset_name", "encoding_circuit_name", "method", "num_layers" ])
            
            print("KEEPING ONLY THE TOP PERFORMING ENTRIES without bandwidth")
            print(f"Expecting number_seed * number_qubits", len(quantum_performance_with_kernels))

            df_poly = df_poly.drop_duplicates(subset=["num_qubits",  "seed", "dataset_name", "kernel", "degree",  ])

            print(f"Expecting number_seed * number_qubits * number_models ", len(df_poly))
            performance_and_kernels_pd = pd.merge(
                df_poly,
                quantum_performance_with_kernels,
                on=["num_qubits", "seed", "dataset_name"],
                suffixes=('_poly', '_quantum')
            )
            print(f"Expecting for top_performance: number_seed * number_qubits * number_models ", len(performance_and_kernels_pd))

        else:
            print("KEEPING ONLY THE TOP PERFORMING ENTRIES for each bandwidth")        
            quantum_performance_with_kernels = quantum_performance_with_kernels.drop_duplicates(subset=["num_qubits", "bandwidth", "seed", "dataset_name", "encoding_circuit_name", "method", "num_layers" ])
            print(f"Expecting number_seed * number_qubits * num_bandwidths", len(quantum_performance_with_kernels))
            df_poly = df_poly.drop_duplicates(subset=["num_qubits", "bandwidth", "seed", "dataset_name", "kernel", "degree",  ])
            print(f"Expecting number_seed * number_qubits * number_models * * num_bandwidths ", len(df_poly))

            performance_and_kernels_pd = pd.merge(
                df_poly,
                quantum_performance_with_kernels,
                on=["num_qubits", "bandwidth", "seed", "dataset_name"],
                suffixes=('_poly', '_quantum')
            )
            print(f"Expecting for top_performance: number_seed * number_qubits * number_models * number_bandwidths ", len(performance_and_kernels_pd))


        

    else:
        if classification_problem:
            try:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "degree", "gamma", "kernel", "num_qubits", "roc_auc_score", "seed"]]
            except:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "gamma", "kernel", "num_qubits", "roc_auc_score", "seed"]]
        else:
            try:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "degree", "gamma", "kernel", "num_qubits", "mse", "seed"]]
            except:
                df_poly = df_poly[["K_train", "bandwidth", "dataset_name", "gamma", "kernel", "num_qubits", "mse", "seed"]]

        performance_and_kernels_pd = pd.merge(
            df_poly,
            df_kernel,
            on=["num_qubits", "bandwidth", "seed", "dataset_name"],
            suffixes=('_poly', '_quantum')
        )


    # Include C_list, every entry in performance_and_kernels_pd will be repeated len(C_list) times

    # Skip every nth entry
    performance_and_kernels_pd = performance_and_kernels_pd.iloc[::skip_every_nth, :]


    def obtain_C_value(row, geo_diff_without_C):
        if geo_diff_without_C:
            return 0
        else:
            if classification_problem:
                return 1/(2*row["C_quantum"])
            return row["alpha_quantum"]


    if difference_option == "geometric":
        try:
            performance_and_kernels_pd[["g_gen_train", "g_tra_train"]] = performance_and_kernels_pd.swifter.apply(
                lambda row: pd.Series(geometric_difference(row["K_train_poly"], row["K_train_quantum"], obtain_C_value(row, geo_diff_without_C))), axis=1)
        except:
            performance_and_kernels_pd[["g_gen_train", "g_tra_train"]] = performance_and_kernels_pd.apply(
                lambda row: pd.Series(geometric_difference(row["K_train_poly"], row["K_train_quantum"], obtain_C_value(row, geo_diff_without_C))), axis=1)

        performance_and_kernels_pd["g_gen_train"] = performance_and_kernels_pd["g_gen_train"].astype(float)
        performance_and_kernels_pd["g_tra_train"] = performance_and_kernels_pd["g_tra_train"].astype(float)

    elif difference_option == "kernel":
        try:
            performance_and_kernels_pd[["Frobenius Difference", "$d_G$", "$d_Gmod$"]] = performance_and_kernels_pd.swifter.apply(
                lambda row: pd.Series(kernel_difference_metric(row["K_train_poly"], row["K_train_quantum"])), axis=1)
        except:
            performance_and_kernels_pd[["Frobenius Difference", "$d_G$", "$d_Gmod$"]] = performance_and_kernels_pd.apply(
                lambda row: pd.Series(kernel_difference_metric(row["K_train_poly"], row["K_train_quantum"])), axis=1)
            
        try:
            performance_and_kernels_pd[["Knorm_poly", "Knorm_quantum"]] = performance_and_kernels_pd.swifter.apply(
                lambda row: pd.Series(K_fro_norms(row["K_train_poly"], row["K_train_quantum"])), axis=1)
        except:
            performance_and_kernels_pd[["Knorm_poly", "Knorm_quantum"]] = performance_and_kernels_pd.apply(
                lambda row: pd.Series(K_fro_norms(row["K_train_poly"], row["K_train_quantum"])), axis=1)

        performance_and_kernels_pd["Frobenius Difference"] = performance_and_kernels_pd["Frobenius Difference"].astype(float)
        performance_and_kernels_pd["$d_G$"] = performance_and_kernels_pd["$d_G$"].astype(float)
        performance_and_kernels_pd["$d_Gmod$"] = performance_and_kernels_pd["$d_Gmod$"].astype(float)
        performance_and_kernels_pd["Knorm_poly"] = performance_and_kernels_pd["Knorm_poly"].astype(float)
        performance_and_kernels_pd["Knorm_quantum"] = performance_and_kernels_pd["Knorm_quantum"].astype(float)
    
    elif difference_option == "both":
        try:
            performance_and_kernels_pd[["Frobenius Difference", "$d_G$", "$d_Gmod$"]] = performance_and_kernels_pd.swifter.apply(
                lambda row: pd.Series(kernel_difference_metric(row["K_train_poly"], row["K_train_quantum"])), axis=1)
            print("Kernel difference calculated")
            performance_and_kernels_pd[["g_gen_train", "g_tra_train"]] = performance_and_kernels_pd.swifter.apply(
                lambda row: pd.Series(geometric_difference(row["K_train_poly"], row["K_train_quantum"], obtain_C_value(row, geo_diff_without_C)  )), axis=1)
            print("Geometric difference calculated")
        except:
            performance_and_kernels_pd[["Frobenius Difference", "$d_G$", "$d_Gmod$"]] = performance_and_kernels_pd.apply(
                lambda row: pd.Series(kernel_difference_metric(row["K_train_poly"], row["K_train_quantum"])), axis=1)
            print("Kernel difference calculated")
            performance_and_kernels_pd[["g_gen_train", "g_tra_train"]] = performance_and_kernels_pd.apply(
                lambda row: pd.Series(geometric_difference(row["K_train_poly"], row["K_train_quantum"],  obtain_C_value(row, geo_diff_without_C)  )), axis=1)
            print("Geometric difference calculated")

        performance_and_kernels_pd["g_gen_train"] = performance_and_kernels_pd["g_gen_train"].astype(float)
        performance_and_kernels_pd["g_tra_train"] = performance_and_kernels_pd["g_tra_train"].astype(float)
        performance_and_kernels_pd["Frobenius Difference"] = performance_and_kernels_pd["Frobenius Difference"].astype(float)
        performance_and_kernels_pd["$d_G$"] = performance_and_kernels_pd["$d_G$"].astype(float)
        performance_and_kernels_pd["$d_Gmod$"] = performance_and_kernels_pd["$d_Gmod$"].astype(float)
        performance_and_kernels_pd["g_gen_train"] = performance_and_kernels_pd["g_gen_train"].astype(float)
        performance_and_kernels_pd["g_tra_train"] = performance_and_kernels_pd["g_tra_train"].astype(float)

    return performance_and_kernels_pd




def calculate_and_write_kernel_difference_from_poly_results_SVC(df_kernel, df_poly, C_list, path_to_save_results, difference_option = "both", df_performance_quantum = None, keep_all_bandwidths = False,  geo_diff_without_C = False, skip_every_nth = 1):
    results_with_g_df = kernel_difference_from_poly_results_SVC(df_kernel, df_poly, C_list, df_performance_quantum, difference_option, keep_all_bandwidths, geo_diff_without_C, skip_every_nth=skip_every_nth)
    #remove K_train_poly, K_train_quantum columns
    results_with_g_df = results_with_g_df.drop(columns=["K_train_poly", "K_train_quantum", "K_test"])
    results_with_g_df.to_feather(path_to_save_results + ".feather")

