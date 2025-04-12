# %%
# %%
#set parent directory as package
import sys
sys.path.append("..")

# %%


# %%
#set parent directory as package
import sys
from data_tools.tools import read_experiment_dic_results, merge_temporary_files, load_feather_folder_as_pd, write_dic_results, rename_kernel
from data_tools.get_dataset import regression_ds_list

import matplotlib
import h5py
import os
from models.concentration_bounds import A_expressibility, haar_frame_potential, subspace_dimension


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import matplotlib.patches as mpatches



pd.pandas.set_option('display.max_columns', None)



from sklearn.metrics import roc_auc_score

from matplotlib import rc

rc('text', usetex=True)



# %%


def entire_preprocessing(kernel_to_compare, performance_of_quantum_simulations, performance_of_classical_simulations, pd_geo_gram, pd_geo_top_results, df_with_kernels, simulation_set_index, dataset_set_index):

    if performance_of_quantum_simulations["dataset_name"].iloc[0] in regression_ds_list:
        classification_problem = False
    else:
        classification_problem = True

    if classification_problem:    
        performance_of_quantum_simulations_simplified_keys = performance_of_quantum_simulations[["dataset_name", "num_qubits", "C", "bandwidth", "roc_auc_score", "y_pred", "y_pred_train", "varK_train", "d", "K_max_eig", "varK_test", "roc_auc_score_train", "eigenvalues",  "encoding_circuit_name", "seed", "ck", "method", "num_datapoints"]]
        performance_of_classical_simulations_simplified_keys = performance_of_classical_simulations[["dataset_name", "num_qubits", "degree", "C", "bandwidth", "roc_auc_score",  "y_pred", "y_pred_train", "varK_train", "d", "K_max_eig",  "d",  "roc_auc_score_train", "eigenvalues",  "kernel", "seed", "ck", "K_mean_train", "num_datapoints"]]

    else:
        performance_of_quantum_simulations_simplified_keys = performance_of_quantum_simulations[["dataset_name", "num_qubits", "alpha", "bandwidth", "mse", "y_pred", "y_pred_train", "varK_train", "d", "K_max_eig", "varK_test", "mse_train", "eigenvalues",  "encoding_circuit_name", "seed", "ck", "method", "num_datapoints"]]
        performance_of_classical_simulations_simplified_keys = performance_of_classical_simulations[["dataset_name", "num_qubits", "degree", "alpha", "bandwidth", "mse",  "y_pred", "y_pred_train", "varK_train", "d", "K_max_eig",  "d",  "mse_train", "eigenvalues",  "kernel", "seed", "ck", "K_mean_train", "num_datapoints"]]

    performance_of_quantum_simulations_simplified_keys = performance_of_quantum_simulations_simplified_keys.drop_duplicates(["num_qubits", "bandwidth", "encoding_circuit_name", "seed", ])
        
    #Separable_rx, IQPLikeCircuit
    circuit_name = performance_of_quantum_simulations["encoding_circuit_name"].iloc[0]
    PQK_or_FQK = performance_of_quantum_simulations["method"].iloc[0]

    performance_of_quantum_simulations_simplified_keys = performance_of_quantum_simulations_simplified_keys[(performance_of_quantum_simulations_simplified_keys["encoding_circuit_name"] == circuit_name)]
    pd_geo_gram_enc = pd_geo_gram[(pd_geo_gram["encoding_circuit_name"] == circuit_name)]
    pd_geo_top_results_enc = pd_geo_top_results[(pd_geo_top_results["encoding_circuit_name"] == circuit_name)]
    #create num_datapoints column that corresponds to the lenght of the eigenvalues column
    pd_geo_gram_enc["num_datapoints"] = pd_geo_gram_enc["eigenvalues"].apply(lambda x: len(x)*1.25)
    pd_geo_top_results_enc["num_datapoints"] = pd_geo_top_results_enc["eigenvalues"].apply(lambda x: len(x)*1.25)

    # %%
    performance_classical_and_quantum = pd.merge(
        performance_of_classical_simulations_simplified_keys,
        performance_of_quantum_simulations_simplified_keys,
        on=["num_qubits", "bandwidth", "seed", "dataset_name", "num_datapoints"],
        suffixes=('_poly', '_quantum')
    )

    # %%
    performance_classical_and_quantum["classical_kernel"] = performance_classical_and_quantum.apply(lambda x: rename_kernel(x, False), axis=1)
    pd_geo_gram_enc["classical_kernel"] = pd_geo_gram_enc.apply(lambda x: rename_kernel(x, False), axis=1)
    pd_geo_top_results_enc["classical_kernel"] = pd_geo_top_results_enc.apply(lambda x: rename_kernel(x, False), axis=1)

    performance_classical_and_quantum["quantum_kernel_name"] = performance_classical_and_quantum.apply(lambda x: rename_kernel(x, True), axis=1)
    pd_geo_gram_enc["quantum_kernel_name"] = pd_geo_gram_enc.apply(lambda x: rename_kernel(x, True), axis=1)
    pd_geo_top_results_enc["quantum_kernel_name"] = pd_geo_top_results_enc.apply(lambda x: rename_kernel(x, True), axis=1)



    performance_classical_and_quantum = performance_classical_and_quantum.drop(columns=["degree"])

    dataset_name_str = performance_classical_and_quantum["dataset_name"].unique()[0]
    encoding_circuit_name_str = performance_classical_and_quantum["quantum_kernel_name"].unique()[0]
    #kernel_type_to_keep = [encoding_circuit_name_str, kernel_to_compare]
    if type(kernel_to_compare) == str:
        kernel_to_compare = [kernel_to_compare] 
    kernel_type_to_keep = ["Quantum", *kernel_to_compare]


    #create folder with name dataset_name_str_encoding_circuit_name_str if it does not exist
    path_to_save = f"./data/plots/{dataset_name_str}_{encoding_circuit_name_str}_{simulation_set_index}_{dataset_set_index}_{kernel_to_compare}/"
    if not os.path.exists(f"./data/plots/{dataset_name_str}_{encoding_circuit_name_str}_{simulation_set_index}_{dataset_set_index}_{kernel_to_compare}"):
        os.makedirs(f"./data/plots/{dataset_name_str}_{encoding_circuit_name_str}_{simulation_set_index}_{dataset_set_index}_{kernel_to_compare}")

    #pop rows with num_qubits = 18
    N_train = len(performance_classical_and_quantum["y_pred_train_poly"][0])

    
    if classification_problem:
        pd_geo_gramm_performance_both = pd.merge(
            performance_classical_and_quantum,
            pd_geo_gram_enc,
            on=["bandwidth", "num_datapoints", "dataset_name", "num_qubits", "seed", "encoding_circuit_name", "classical_kernel", "quantum_kernel_name", "C_poly", "C_quantum", "kernel", "method"],
            suffixes=('', '_extra')
        )
    else:
        pd_geo_gramm_performance_both = pd.merge(
            performance_classical_and_quantum,
            pd_geo_gram_enc,
            on=["bandwidth", "num_datapoints", "dataset_name", "num_qubits", "seed", "encoding_circuit_name", "classical_kernel", "quantum_kernel_name", "alpha_poly", "alpha_quantum", "kernel", "method"],
            suffixes=('', '_extra')
        )

    print(pd_geo_gramm_performance_both.keys())
    #Cleaning because I am not merging on continous variables, which are also equal (i.e. roc_auc_score_poly vs roc_auc_score_poly_extra) )
    columns_to_drop = [col for col in pd_geo_gramm_performance_both.columns if col.endswith('_extra')]
    pd_geo_gramm_performance_both = pd_geo_gramm_performance_both.drop(columns=columns_to_drop)


    if classification_problem:
        indices_of_best_quantum = pd_geo_gramm_performance_both.groupby(["num_qubits", "classical_kernel", "quantum_kernel_name", "seed"])["roc_auc_score_quantum"].idxmax()
        indices_of_best_poly = pd_geo_gramm_performance_both.groupby(["num_qubits", "classical_kernel", "quantum_kernel_name", "seed"])["roc_auc_score_poly"].idxmax()

    else:
        indices_of_best_quantum = pd_geo_gramm_performance_both.groupby(["num_qubits", "classical_kernel", "quantum_kernel_name", "seed"])["mse_quantum"].idxmin()
        indices_of_best_poly = pd_geo_gramm_performance_both.groupby(["num_qubits", "classical_kernel", "quantum_kernel_name", "seed"])["mse_poly"].idxmin()
    values_of_best_quantum = pd_geo_gramm_performance_both.loc[indices_of_best_quantum]
    values_of_best_quantum = values_of_best_quantum.sort_values(by="num_qubits")


    values_of_best_poly = pd_geo_gramm_performance_both.loc[indices_of_best_quantum]
    values_of_best_poly = values_of_best_poly.sort_values(by="num_qubits")


    values_of_best_quantum["mse_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: np.mean( (row["y_pred_poly"] - row["y_pred_quantum"])**2  ) , axis=1)
    values_of_best_quantum["mse_train_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: np.mean( (row["y_pred_train_poly"] - row["y_pred_train_quantum"])**2  ) , axis=1)
    values_of_best_quantum["VarK_train_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: np.abs(row["varK_train_poly"] - row["varK_train_quantum"])/row["varK_train_quantum"], axis=1)
    values_of_best_quantum["VarK_train_divided_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: np.abs(row["varK_train_poly"])/row["varK_train_quantum"]-1, axis=1)




    def _roc_auc_score(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return np.nan
        
    if classification_problem:
        values_of_best_quantum["roc_auc_score_between_poly_quantum"] = values_of_best_quantum.apply(lambda row:  _roc_auc_score(row["y_pred_poly"], row["y_pred_quantum"]) , axis=1)
        values_of_best_quantum["roc_auc_score_train_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: _roc_auc_score(row["y_pred_train_poly"], row["y_pred_train_quantum"]) , axis=1)

        values_of_best_poly["roc_auc_score_between_poly_quantum"] = values_of_best_quantum.apply(lambda row:  _roc_auc_score(row["y_pred_poly"], row["y_pred_quantum"]) , axis=1)
        values_of_best_poly["roc_auc_score_train_between_poly_quantum"] = values_of_best_quantum.apply(lambda row: _roc_auc_score(row["y_pred_train_poly"], row["y_pred_train_quantum"]) , axis=1)




    # %%
    
    #merge df_with_kernels with pd_geo_gramm_performance_both on [ "encoding_circuit_name", "num_qubits", "expressivity", "dataset_name", "method", "seed", "num_datapoints", "num_layers"]
    pd_geo_gramm_performance_both = pd.merge(
        pd_geo_gramm_performance_both,
        df_with_kernels,
        on=[ "encoding_circuit_name", "num_qubits", "dataset_name", "method", "seed", "num_datapoints", "num_layers", "bandwidth"],
        suffixes=('', '_extra')
    )
    
    return pd_geo_gramm_performance_both, pd_geo_top_results_enc, values_of_best_quantum, values_of_best_poly, N_train, dataset_name_str, encoding_circuit_name_str, kernel_type_to_keep, path_to_save



def get_expressibility_from_df(df_with_kernels, simulation_set_index, dataset_set_index, root_path):
    PQK_to_FQK_dict = {"0_9":"0_1", "0_10":"0_2", "0_11":"0_3", "0_12":"0_4", "0_13":"0_5", "0_14":"0_6", "0_15":"0_7", "0_16":"0_8", "0_26":"0_25", "0_28":"0_27", "0_32":"0_30", "0_33":"0_31"}
    set_dataset_str = f"{simulation_set_index}_{dataset_set_index}"
    if set_dataset_str in PQK_to_FQK_dict: #if True, we are using PQK but need to use FQK for the expressibility
        df_with_kernels_FQK = pd.DataFrame(read_experiment_dic_results(f"{root_path}kernels_{PQK_to_FQK_dict[set_dataset_str]}.h5", ignore_Ks=False, short_load=False))
        #sort df_with_kernels_FQK by num_qubits, seed, bandwidth, dataset_name
        df_with_kernels_FQK = df_with_kernels_FQK.sort_values(by=["num_qubits", "seed", "bandwidth", "dataset_name", "num_layers"])
        df_with_kernels_FQK = df_with_kernels_FQK.reset_index(drop=True)
        df_with_kernels = df_with_kernels.sort_values(by=["num_qubits", "seed", "bandwidth", "dataset_name", "num_layers"])
        df_with_kernels = df_with_kernels.reset_index(drop=True)
        df_with_kernels["expressivity"] = df_with_kernels_FQK.apply(lambda row: A_expressibility(row["K_train"], row["num_qubits"], t=2), axis=1)


        print("keys OF FQKs")
        print(df_with_kernels_FQK.keys())
        print("keys OF PQKs")
        print(df_with_kernels.keys())
        print("-+++++++++++++++++++++++++++++++")
        print("HEAD OF FQKs")
        print(df_with_kernels_FQK[["num_qubits", "seed", "bandwidth", "dataset_name", "num_layers"]].head())

        print("HEAD OF PQKs")
        print(df_with_kernels[["num_qubits", "seed", "bandwidth", "dataset_name", "num_layers"]].head())
    else:
        df_with_kernels["expressivity"] = df_with_kernels.apply(lambda row: A_expressibility(row["K_train"], row["num_qubits"], t=2), axis=1)
    return df_with_kernels

# %%
def load_and_return_preprocessed(simulation_set_index, dataset_set_index, classical_name, kernel_to_compare, add_extra_path=False):
    print(f"Simulation set index: {simulation_set_index} and dataset set index: {dataset_set_index}")

    root_path = "./data/results/"
    if add_extra_path:
        root_path = "../data/results/"

    performance_of_classical_simulations = load_feather_folder_as_pd(f"{root_path}{classical_name}_classical/")
    #performance_of_classical_simulations.drop(columns=['encoding_circuit_name', 'method'], inplace=True)

    #performance_of_quantum_simulations = load_feather_folder_as_pd(f"./data/results/svc_performance_quantum_{simulation_set_index}_{dataset_set_index}")
    performance_of_quantum_simulations = load_feather_folder_as_pd(f"{root_path}svc_performance_quantum_{simulation_set_index}_{dataset_set_index}", initial_key="performance_quantum")
    pd_geo_gram = pd.read_feather(f"{root_path}svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}.feather")
    pd_geo_top_results = pd.read_feather(f"{root_path}svc_performance_quantum_{simulation_set_index}_{dataset_set_index}/"+ f"Vboth_difference_{simulation_set_index}_{dataset_set_index}_top_results.feather")


    df_with_kernels = pd.DataFrame(read_experiment_dic_results(f"{root_path}kernels_{simulation_set_index}_{dataset_set_index}.h5", ignore_Ks=False, short_load=False))
    df_with_kernels = get_expressibility_from_df(df_with_kernels, simulation_set_index, dataset_set_index, root_path)
    df_with_kernels = df_with_kernels[[ "encoding_circuit_name", "num_qubits", "expressivity", "dataset_name", "method", "seed", "num_datapoints", "num_layers", "bandwidth"]]


    pd_geo_gramm_performance_both, pd_geo_top_results_enc, values_of_best_quantum, values_of_best_poly, N_train, dataset_name_str, encoding_circuit_name_str, kernel_type_to_keep, path_to_save = entire_preprocessing(kernel_to_compare, performance_of_quantum_simulations, performance_of_classical_simulations, pd_geo_gram, pd_geo_top_results, df_with_kernels, simulation_set_index, dataset_set_index)

    return pd_geo_gramm_performance_both, pd_geo_top_results_enc, values_of_best_quantum, values_of_best_poly, N_train, dataset_name_str, encoding_circuit_name_str, kernel_type_to_keep, path_to_save

# %%
def top_enc_melting_preprocess(pd_geo_top_results_enc, kernel_type_to_keep, encoding_circuit_name_str, kernel_to_compare, names_of_metrics, use_kernel_acronym=False):
    kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str = names_of_metrics

    if "roc_auc_score_quantum" in pd_geo_top_results_enc.keys():
        classification_problem = True
        performance_keys = ["roc_auc_score_quantum", "roc_auc_score_poly"]
    else:
        classification_problem = False
        performance_keys = ["mse_quantum", "mse_poly"]
    print("classification_problem", classification_problem)

    

    metrics = {
        performance_str: performance_keys,
        top_eigenvalue_str: ["K_max_eig_quantum", "K_max_eig_poly"],
        bandwidth_str: ["top_bandwidth_quantum", "top_bandwidth_poly"],
        g_str: ["g_gen_train"],
        frobenius_str: ["Frobenius Difference"], #$d_Gmod$, $d_G$ Frobenius Difference
        #varK_str: ["varK_train_poly", "varK_train_quantum"]
    }

    # Filter the data

    final_array_ = pd_geo_top_results_enc.copy()
    #final_array_ = pd_geo_top_results_enc_varK.copy()

    final_array_["top_bandwidth_quantum"] = final_array_["bandwidth_quantum"]
    final_array_["top_bandwidth_poly"] = final_array_["bandwidth_poly"]


    #final_array_ = final_array_[final_array_["classical_kernel"] == "poly_3"]

    # Melt the data to long format
    melted = pd.melt(final_array_,
                    id_vars=[ "classical_kernel", "num_qubits", "seed"],
                    value_vars=sum(metrics.values(), []),
                    var_name="metric_type",
                    value_name="value")

    #add a new column called kernel_type which is a copy of classical_kernel
    melted["kernel_type"] = melted["classical_kernel"]
    #now, if "quantum" in the metric_type, then we change kernel_type to Quantum
    melted["kernel_type"] = melted.apply(lambda x: "Quantum" if "quantum" in x["metric_type"] else x["kernel_type"], axis=1)
    melted.drop_duplicates(["num_qubits", "seed", "metric_type",  "kernel_type"], inplace=True) #Previously, "value" was here. Removing it is necessary for it to work


    #only keep rows with kernel_type Quantum, poly_3, poly_4,

    melted = melted[melted["kernel_type"].isin(kernel_type_to_keep)]
    #make kernel_type a category with order Quantum, poly_3, 
    melted["kernel_type"] = pd.Categorical(melted["kernel_type"], kernel_type_to_keep)
    #sort melted by num_qubits and kernel_type

    #change if metric_type is bandwidth_str, then change kernel_type to Quantum


    # Add a new column to distinguish between different categories of metrics
    def get_metric_category(metric_type):
        for category, metric_list in metrics.items():
            if metric_type in metric_list:
                return category
        return "Other"

    melted["metric_category"] = melted["metric_type"].apply(get_metric_category)

    melted["kernel_type"] = melted.apply(lambda x: "Quantum" if x["metric_category"] == g_str else x["kernel_type"], axis=1)
    melted["kernel_type"] = melted.apply(lambda x: "Quantum" if x["metric_category"] == frobenius_str else x["kernel_type"], axis=1)
    if use_kernel_acronym == False:
        melted["kernel_type"] = melted.apply(lambda x: encoding_circuit_name_str if x["kernel_type"] == "Quantum" else x["kernel_type"], axis=1)
    else:
        melted["kernel_type"] = melted.apply(lambda x: encoding_circuit_name_str[-3:] if x["kernel_type"] == "Quantum" else x["kernel_type"], axis=1)
        #rename FQK to $k^{\mathrm{FQK}}$ and PQK to $k^{\mathrm{PQK}}$

    #melted["kernel_type"] = pd.Categorical(melted["kernel_type"], kernel_type_to_keep)

    kernel_type_to_keep_du = [encoding_circuit_name_str, kernel_to_compare]
    melted[kernel_type_str] = melted["kernel_type"]


    num_qubit_order = melted["num_qubits"].unique()
    num_qubit_order.sort()
    return melted, kernel_type_to_keep_du, num_qubit_order
        

sns.set_context("paper", rc={"font.size":14,"axes.titlesize":16,"axes.labelsize":22, 
                             "legend.fontsize":16, "xtick.labelsize":18, 
                             "ytick.labelsize":18, "legend.title_fontsize":14,  'lines.linewidth': 3, 'lines.markersize': 6, })


kernel_type_str = "Kernel "
num_qubit_str = "$\#$ dimensions $n$"
performance_str = "roc auc score"
top_eigenvalue_str = r"$\eta_{\mathrm{max}}$"
frobenius_str = r"$F(\mathbf{K}_{\mathrm{C}}, \mathbf{K}_{\mathrm{Q}})$"
g_str = r"$g(\mathbf{K}_{\mathrm{C}}, \mathbf{K}_{\mathrm{Q}})$"
bandwidth_str = "bandwidth $c^*$"
varK_str = r"Var$_{\mathcal{D}}[\mathbf{K}]$"
expressibity_str = r"$\epsilon_{\mathbb{U}_{\mathcal{X}}}$"


def kernel_to_compare_to_latex_str(kernel_to_compare):
    if kernel_to_compare.startswith("poly") or kernel_to_compare.startswith("rbf_poly"):
        return "$k_{\mathrm{cl}}^{(" + kernel_to_compare[-1] + ")}$"
    elif kernel_to_compare.startswith("rbf"):
        return "$k_{\mathrm{cl}}^{(\mathrm{RBF})}$"
    elif kernel_to_compare.startswith("FQK"):
        return "$k^{(\mathrm{FQK})}$"
    elif kernel_to_compare.startswith("PQK"):
        return "$k^{(\mathrm{PQK})}$"
    else:
        return kernel_to_compare


ci = 0.68
def ci95_low(series):
    import scipy.stats as stats
    #stats.norm.interval(0.68, loc=mu, scale=sigma/sqrt(N))
    mean = series.mean()
    std_err = series.sem()  # Standard error of the mean
    return stats.norm.interval(ci, loc=mean, scale=std_err)[0]
def ci95_high(series):
    import scipy.stats as stats
    #stats.norm.interval(0.68, loc=mu, scale=sigma/sqrt(N))
    mean = series.mean()
    std_err = series.sem()  # Standard error of the mean
    return stats.norm.interval(ci, loc=mean, scale=std_err)[1]



def melted_preprocess_as_of_bandwidth(pd_geo_gramm_performance_both, kernel_to_compare, num_qubit_order, encoding_circuit_name_str, names_of_metrics):

    kernel_type_str, num_qubit_str, performance_str, top_eigenvalue_str, frobenius_str, g_str, bandwidth_str, varK_str, expressibity_str = names_of_metrics
    ######################################################################################################

    if type(kernel_to_compare) == str:
        kernel_to_compare = [kernel_to_compare]

    #determine if classification or regression
    if "roc_auc_score_quantum" in pd_geo_gramm_performance_both.keys():
        classification_problem = True
        performance_keys = ["roc_auc_score_quantum", "roc_auc_score_poly"]
    else:
        classification_problem = False
        performance_keys = ["mse_quantum", "mse_poly"]
    
    metrics = {
        performance_str: performance_keys,
        varK_str: ["varK_train_quantum", "varK_train_poly"],
        top_eigenvalue_str: ["K_max_eig_poly", "K_max_eig_quantum"],
        frobenius_str: ["Frobenius Difference"],
        g_str: ["g_gen_train"],
        expressibity_str: ["expressivity"],
    }


    # Filter the data
    final_array_ = pd_geo_gramm_performance_both.copy()
    final_array_ = final_array_[~final_array_["classical_kernel"].str.startswith("taylor")]

    # Melt the data to long format
    melted = pd.melt(final_array_,
                    id_vars=["bandwidth", "num_qubits", "classical_kernel", "seed"],
                    value_vars=sum(metrics.values(), []),
                    var_name="metric_type",
                    value_name="value")



    # Add a new column to distinguish between different categories of metrics
    def get_metric_category(metric_type):
        for category, metric_list in metrics.items():
            if metric_type in metric_list:
                return category
        return "Other"

    melted["metric_category"] = melted["metric_type"].apply(get_metric_category)

    # Add a new column to distinguish between quantum and poly metrics
    melted["metric_style"] = melted["metric_type"].apply(lambda x: 'quantum' if 'quantum' in x else 'classical')
    melted_only_comparedkernel = melted[melted["classical_kernel"].isin(kernel_to_compare)]
    print(melted_only_comparedkernel["classical_kernel"].unique())
    melted_only_comparedkernel["num_qubits"] = pd.Categorical(melted_only_comparedkernel["num_qubits"], categories=num_qubit_order, ordered=True)
    melted_only_comparedkernel["kernel_type"] = melted_only_comparedkernel.apply(lambda x: "Quantum" if "quantum" in x["metric_style"] else x["classical_kernel"], axis=1)
    print(melted_only_comparedkernel["kernel_type"].unique())

    melted_only_comparedkernel[kernel_type_str] = melted_only_comparedkernel["kernel_type"]
    melted_only_comparedkernel[kernel_type_str] = melted_only_comparedkernel.apply(lambda x: encoding_circuit_name_str if x[kernel_type_str] == "Quantum" else x[kernel_type_str], axis=1)

    melted_only_comparedkernel[num_qubit_str] = melted_only_comparedkernel["num_qubits"]
    melted_only_comparedkernel[bandwidth_str] = melted_only_comparedkernel["bandwidth"]


    #TODO Find more elegant way of doing this
    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == expressibity_str, "metric_style"] = "quantum"
    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == expressibity_str, kernel_type_str] = encoding_circuit_name_str

    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == frobenius_str, "metric_style"] = "quantum"
    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == frobenius_str, kernel_type_str] = encoding_circuit_name_str

    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == g_str, "metric_style"] = "quantum"
    melted_only_comparedkernel.loc[melted_only_comparedkernel["metric_category"] == g_str, kernel_type_str] = encoding_circuit_name_str

    #sepia progressive
    color_palette = sns.color_palette("rocket_r", n_colors=len(num_qubit_order) )

    melted_only_comparedkernel_dummy = melted_only_comparedkernel.copy()
    #remove all performance_str from melted_only_comparedkernel_dummy
    melted_only_comparedkernel_dummy = melted_only_comparedkernel_dummy[melted_only_comparedkernel_dummy["metric_category"] != performance_str]

    #sort by num_qubits
    melted_only_comparedkernel_dummy = melted_only_comparedkernel_dummy.sort_values(by=["num_qubits"])
                                
    # Define the unique values for rows and hues
    metric_categories = [varK_str, top_eigenvalue_str, expressibity_str, frobenius_str, g_str]
    num_qubit_order = sorted(melted_only_comparedkernel_dummy[num_qubit_str].unique())
    kernel_types = melted_only_comparedkernel_dummy[kernel_type_str].unique()
    return melted_only_comparedkernel_dummy, metric_categories, num_qubit_order, kernel_types, color_palette 

def plot_metric_bandwidth_different_plotting_styles(pd_geo_top_results_enc, pd_geo_gramm_performance_both, kernel_type_to_keep, encoding_circuit_name_str, dataset_name_str, path_to_save, N_train, kernel_to_compare, num_qubit_order, names_of_metrics, hide_ylabel=False, show_plt = False, single_param_style = "full", extra_number = 2, only_classical = False):
    
        
    
    sns.set_context("paper", rc={"font.size":12,"axes.titlesize":16,"axes.labelsize":22, 
                                "legend.fontsize":12, "xtick.labelsize":18, 
                                "ytick.labelsize":18, "legend.title_fontsize":12,  'lines.linewidth': 3, 'lines.markersize': 6, })

    plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    
    melted_only_comparedkernel_dummy, metric_categories, num_qubit_order, kernel_types, color_palette = melted_preprocess_as_of_bandwidth(pd_geo_gramm_performance_both, kernel_to_compare, num_qubit_order, encoding_circuit_name_str, names_of_metrics)

    # include and extra axis to remove it to have proper height rations
    if single_param_style == "full":
        fig, axes = plt.subplots(len(metric_categories)+1, 1, figsize=(4.5, 16), sharex=True, gridspec_kw= {'height_ratios': [1, 1, 1, 1, 1, 1]})
    elif single_param_style == "single":
        fig, axes = plt.subplots(3+2, 1, figsize=(4.5, 11), sharex=True, gridspec_kw= {'height_ratios': [1, 1, 1, 0.2, 1]})
    elif single_param_style == "single_without" or single_param_style == "double_g":
        fig, axes = plt.subplots(3, 1, figsize=(4.5, 7), sharex=True, gridspec_kw= {'height_ratios': [1, 1, 1,]})
    elif single_param_style == "double" or single_param_style == "individual_without":
        fig, axes = plt.subplots(2, 1, figsize=(4.5, 5), sharex=True, gridspec_kw= {'height_ratios': [1, 1]})
    elif single_param_style == "double_g_F":
        fig, axes = plt.subplots(4, 1, figsize=(4.5, 9), sharex=True, gridspec_kw= {'height_ratios': [1, 1, 1, 1]})
    elif single_param_style == "no_expressibility":
        fig, axes = plt.subplots(5, 1, figsize=(4.5, 14), sharex=True, gridspec_kw= {'height_ratios': [1, 1, 1, 1, 1]})




    def kernel_to_compare_to_latex_str(kernel_to_compare):
        if kernel_to_compare.startswith("poly") or kernel_to_compare.startswith("rbf_poly"):
            return "$k^{(" + kernel_to_compare[-1] + ")}$"
        elif kernel_to_compare.startswith("rbf"):
            return "$k^{(\mathrm{RBF})}$"
        elif kernel_to_compare.startswith("FQK"):
            return "$k^{(\mathrm{FQK})}$"
        elif kernel_to_compare.startswith("PQK"):
            return "$k^{(\mathrm{PQK})}$"
        elif ("FQK") in kernel_to_compare:
            return "$k^{(\mathrm{FQK})}$"
        elif ("PQK") in kernel_to_compare:
            return "$k^{(\mathrm{PQK})}$"

    ax = axes[-1]
    melted_only_optimal = top_enc_melting_preprocess(pd_geo_top_results_enc, kernel_type_to_keep, encoding_circuit_name_str, kernel_to_compare, names_of_metrics[:-1])[0]
    melted_only_optimal = melted_only_optimal[melted_only_optimal["metric_category"] == bandwidth_str]
    melted_only_optimal[num_qubit_str] = melted_only_optimal["num_qubits"]
    kernel_to_compare = kernel_to_compare[0] if type(kernel_to_compare) == list else kernel_to_compare

    #melted_only_optimal[kernel_type_str] = melted_only_optimal.apply(lambda x: kernel_to_compare_to_latex_str(x[kernel_type_str]) if x[kernel_type_str] == kernel_to_compare else x["kernel_type"], axis=1)
    melted_only_optimal[kernel_type_str] = melted_only_optimal.apply(lambda x: kernel_to_compare_to_latex_str(x[kernel_type_str]) , axis=1)


    if single_param_style == "full" or single_param_style == "single" or single_param_style =="no_expressibility":
        sns.boxplot(data=melted_only_optimal, x="value", y=num_qubit_str, hue=kernel_type_str, palette='colorblind', ax=ax, orient='h', flierprops={"marker": "d", "markerfacecolor": "black"})
        print(melted_only_optimal[kernel_type_str])

        text = "(F"
        text += "1)" if hide_ylabel == False else f"{extra_number})"

    
    if single_param_style == "full" or single_param_style == "single" or single_param_style =="no_expressibility":
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', horizontalalignment='left')

    #change position of legend of ax to the lower left
    if hide_ylabel == False:
        pass
    else:
        #remove ax y label
        ax.set_yticklabels([])
        ax.set_ylabel("")
    
    if single_param_style == "full" or single_param_style == "single" or single_param_style =="no_expressibility":
        ax.legend(loc='lower right', title="Optimal $c^*$", fontsize=13, title_fontsize=13, handletextpad=0.3, bbox_to_anchor=(1.01,-0.03))        
    
    num_qubits_to_keep = [2, 6, 12, 16]
    solid_lines_for_legend = []
    num_qubit_order = num_qubit_order[:-1]
    color_palette = sns.color_palette("rocket_r", n_colors=len(num_qubits_to_keep) )
    color_palette_dictio = {n_q: color for n_q, color in zip(num_qubits_to_keep, color_palette)}

    melted_only_comparedkernel_dummy = melted_only_comparedkernel_dummy[melted_only_comparedkernel_dummy["num_qubits"].isin(num_qubit_order)]
    ax.set_xscale("log")
    counter = 0
    kernel_type_label_str = []
    # Loop through each metric category and plot
    if single_param_style == "full":
        metric_categories = [varK_str, top_eigenvalue_str, expressibity_str, g_str, frobenius_str, ]
    elif single_param_style == "single" :
        metric_categories = [varK_str, top_eigenvalue_str, expressibity_str]
    elif single_param_style == "double":
        metric_categories = [g_str, frobenius_str]
    elif single_param_style == "double_g":
        metric_categories = [varK_str, top_eigenvalue_str, g_str]
    elif single_param_style == "double_g_F":
        metric_categories = [varK_str, top_eigenvalue_str, g_str, frobenius_str]
    elif single_param_style == "individual_without":
        metric_categories = [varK_str, top_eigenvalue_str]
    elif single_param_style == "no_expressibility":
        metric_categories = [varK_str, top_eigenvalue_str, g_str, frobenius_str]

    
    for i, metric in enumerate(metric_categories):
        ax = axes[i]
        for kernel_type in kernel_types:
            for nq_idx, num_qubits in enumerate(num_qubit_order):
                subset = melted_only_comparedkernel_dummy[(melted_only_comparedkernel_dummy["metric_category"] == metric) &
                                                        (melted_only_comparedkernel_dummy[num_qubit_str] == num_qubits) &
                                                        (melted_only_comparedkernel_dummy[kernel_type_str] == kernel_type)]
                
                # Group by bandwidth and calculate mean and std and 95 CI
                grouped = subset.groupby(bandwidth_str)["value"].agg(['mean', 'std', ci95_low, ci95_high]).reset_index()
                
                # Plot mean line

                if "FQK" in kernel_type or "PQK" in kernel_type:
                    style = 'solid'
                    if only_classical:
                        break 
                elif kernel_type.startswith("rbf_poly") or kernel_type.startswith("rbf"):
                    style = 'dashed'
                
                if metric == frobenius_str or metric == g_str:
                    style = 'dashdot'
                    #y ticks label 250, 500, 750, 
                    ax.set_yticks([0, 250, 500, 750])

                if kernel_to_compare == kernel_type and metric == g_str and hide_ylabel == False and num_qubits in num_qubit_order:
                    label_str = f"{num_qubits}"
                    style = 'dashdot'
                elif metric == varK_str:
                    if nq_idx == len(num_qubit_order) - 1:
                        label_str = f"{kernel_to_compare_to_latex_str(kernel_type)}"
                        kernel_type_label_str.append(label_str)
                    else:
                        label_str = None
                else:
                    label_str = None

                #if num_qubits is nan break
                

                color = color_palette_dictio[num_qubits]
                ax.plot(grouped[bandwidth_str], grouped['mean'], c=color, label=label_str, linestyle=style)

                # Plot error bars

                # if kernel_type == encoding_circuit_name_str:
                #     ax.fill_between(grouped[bandwidth_str], grouped['ci95_low'], grouped['ci95_high'], alpha=0.3, color=color_palette[nq_idx])

                # Plot shaded area for 95% CI
                if kernel_type != kernel_to_compare:
                    pass
                    #ax.fill_between(grouped[bandwidth_str], grouped['ci95_low'], grouped['ci95_high'], alpha=0.3)               
                # Scatter points
                #ax.scatter(grouped[bandwidth_str], grouped['mean'], s=2)
        row_val = metric
        if row_val == performance_str:             
            ax.set_ylim(0.45, 0.8)
        if row_val == varK_str:
            text = "(A"
            text += "1)" if hide_ylabel == False else f"{extra_number})"

            #remove duplicates from kernel_type_label_str
            kernel_type_label_str = list(set(kernel_type_label_str))
            #axes[0].set_title(f"{kernel_type_label_str[0]} vs {kernel_type_label_str[1]}")
            ax.set_yscale("log")
            ax.set_ylim(10**-11, 10**5)
            #remove legend
            ax.set_yticks([10**i for i in range(-9, 4, 2)])
            #if hide_ylabel == False:
            #title for the plot
            handles, labels = ax.get_legend_handles_labels()
            # Modify the first label
            if "FQK" in labels[0] or "PQK" in labels[0]:
                pass 
            else:
                #reverse labels
                labels = labels[::-1]
                handles = handles[::-1]
            if only_classical:
                pass
            else:      
                labels[0] += " and"
            
            
            if only_classical == False:
                kernel_type_legend = ax.legend(
                    handles, labels,
                    loc='upper center', 
                    fontsize=20, 
                    handletextpad=0.3, 
                    bbox_to_anchor=(0.5, 1.24),  # Adjust this value to align with the title
                    ncol=2,  # Optional: Spread legend items horizontally, #remove
                    columnspacing=0.5,  # Optional: Add horizontal space between columns
                    frameon=False

                )
            else:
                kernel_type_legend = ax.legend(
                    handles, labels,
                    loc='upper center', 
                    fontsize=20, 
                    handletextpad=0.3, 
                    bbox_to_anchor=(0.5, 1.3),  # Adjust this value to align with the title
                    ncol=2,  # Optional: Spread legend items horizontally, #remove
                    columnspacing=0.5,  # Optional: Add horizontal space between columns
                    frameon=False
                )
                


            manual_patches = []
            for label, color in color_palette_dictio.items():
                if label in num_qubits_to_keep:
                    manual_patches.append(mpatches.Patch(color=color, label=label))
                                            
            # Second legend for the color mapping
            if hide_ylabel == False:
                color_legend = ax.legend(handles=manual_patches, 
                                        loc="center", 
                                        title=num_qubit_str,
                                        handleheight=0.05,  # Reduce height of handles
                                        handlelength=1,  # Reduce length of legend handles
                                        columnspacing=0.5,  
                                        ncol=4,
                                        handletextpad=0.3,
                                        fontsize=14,
                                        title_fontsize=14, #put the label on the top center of the plot
                                        bbox_to_anchor=(0.55, 0.83),
                                        )
            # Add the first legend back to the plot
            ax.add_artist(kernel_type_legend)
            #include text A1 on the top left of the plot
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')
        elif row_val == performance_str:
            pass
        elif row_val == top_eigenvalue_str:
            text = "(B"
            text += "1)" if hide_ylabel == False else f"{extra_number})"

            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')

            ax.set_yscale("log")
            ax.set_yticks([10**i for i in range(-2, 1, 1)])
            ax.set_ylim(10**-3, 10)
            #ax.axhline(1/(N_train), color="black", linestyle="--", label="1/N") 
            #best location for legend , legend_fontsize = 14, and fonttitle =
        elif row_val == expressibity_str:
            text = "(C"
            text += "1)" if hide_ylabel == False else f"{extra_number})"

            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')

            ax.set_yscale("log")
            ax.set_ylim(0.5*10**-2, 5.5)
            ax.set_yticks([10**i for i in range(-1, 1, 1)])

        elif row_val == frobenius_str:
            #ax.set_yscale("log")
            text = "(E"
            text += "1)" if hide_ylabel == False else f"{extra_number})"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')


            if kernel_to_compare == "rbf":
                ax.set_yscale("log")
                ax.set_ylim(10**-3, 10**3)
            else:
                ax.set_yscale("log")
                ax.set_ylim(10**-3, 10**3)
        elif row_val == g_str:
            text = "(D"
            text += "1)" if hide_ylabel == False else f"{extra_number})"

            
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')


            label_str_g_str = "$\sqrt{N}$" if hide_ylabel == False else None
            if kernel_to_compare == "rbf":
                ax.set_yscale("log")
                ax.set_ylim(0.5*10**0, 10**3)
            else:
                ax.set_yscale("log")
                ax.set_ylim(0.5*10**0, 10**3)
            #if hide_ylabel == False:
            #horizontal_legend = ax.legend(loc='lower right')
            if hide_ylabel == False:
                #hue_legend = ax.legend(loc='upper left', title=num_qubit_str, ncols=2)
                ax.text(0.95, 0.60, label_str_g_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right')

            #clean up legend
            #ax.add_artist(hue_legend)
            ax.axhline(np.sqrt(N_train), color="black", linestyle=":", label=label_str_g_str) 
            #put a text label on the right side of the plot
            
            #horizontal_legend = ax.legend(loc='upper right')
            #ax.add_artist(horizontal_legend)
        else:
            pass
        ax.set_xscale("log")
        #ax.set_title("")
        ax.set_xticks([10**i for i in range(-4, 2, 1)])
        

        
        counter += 1
        if hide_ylabel == False:
            ax.set_ylabel(metric)
        else:
            ax.set_yticklabels([])
        #ax.set_yscale('log')
        ax.set_xscale('log')

    if single_param_style == "double":
        axes[-1].set_xlabel("Bandwidth $c$", fontsize=18)
    elif single_param_style == "individual_without":
        axes[1].set_xlabel("Bandwidth $c$", fontsize=18)
    else:
        #axes[-1].set_xlabel("Bandwidth $c^*$")
        axes[-1].set_xlabel("Bandwidth $c$", fontsize=22)
    
    axes[0].set_title(f"{dataset_name_str}", color="white", zorder=0, fontsize=21)

    #reduce vertical separation between subplots
    plt.subplots_adjust(hspace=0.06)

    # if single_param_style != "double":
    #     axes[-2].axis('off')

    if only_classical:
        single_param_style += "_classical"
    #plt.tight_layout()
    #plt savefig 
    #plt.figure(constrained_layout=True)
    fig.align_ylabels(axes)
    if show_plt:
        plt.show()
    else:     
        if hide_ylabel:
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_{single_param_style}_quantities_vs_bandwidth_cQ_and_cC_{kernel_to_compare}_bandwidth_hidden.pdf", dpi=500,  bbox_inches = 'tight')
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_{single_param_style}_quantities_vs_bandwidth_cQ_and_cC_{kernel_to_compare}_bandwidth_hidden.png", dpi=500,  bbox_inches = 'tight')
        else:
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_{single_param_style}_quantities_vs_bandwidth_cQ_and_cC_{kernel_to_compare}.pdf", dpi=500,  bbox_inches = 'tight')
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_{single_param_style}_quantities_vs_bandwidth_cQ_and_cC_{kernel_to_compare}.png", dpi=500,  bbox_inches = 'tight')
            


def plot_metric_numdimension_normal(melted, encoding_circuit_name_str, dataset_name_str, path_to_save, N_train, kernel_to_compare, hide_xlabels=False, show_plt=False, show_abc=False):
    #import gridspec
    import matplotlib.gridspec as gridspec

    color_palette = sns.color_palette("colorblind")
    melted_copy = melted.copy()

    melted_copy["kernel_type"] = melted_copy.apply(lambda x: kernel_to_compare_to_latex_str(x["kernel_type"]), axis=1)
    melted_copy["Kernel "] = melted_copy.apply(lambda x: kernel_to_compare_to_latex_str(x["Kernel "]), axis=1)
    kernel_to_compare_in_proper_latex = [kernel_to_compare_to_latex_str(i) for i in kernel_to_compare]
    hue_order = [kernel_to_compare_to_latex_str("FQK"), kernel_to_compare_to_latex_str("PQK"), *kernel_to_compare_in_proper_latex]

    # Create a box plot for each metric category
    metric_categories = [performance_str, bandwidth_str, top_eigenvalue_str, g_str, frobenius_str, ]
    num_categories = len(metric_categories)


    if type(kernel_to_compare) == str:
        kernel_to_compare = [kernel_to_compare]

    # Create a 2x2 grid but let ax1_top span both rows
    #decrease the widh separation
    fig, axs_list = plt.subplots(1, num_categories, figsize=(num_categories * 5.5, 4.5),gridspec_kw={  'wspace': 0.3})  
    # fig = plt.figure(figsize=(num_categories * 5.5, 4.25))
    # gs = gridspec.GridSpec(1, num_categories, width_ratios=[1, 0.5, 1, 1, 0.5])  # Adjust gaps

    # axs_list = [fig.add_subplot(gs[i]) for i in range(num_categories)]

    combined_str = "combined"
    
    for i in range(len(metric_categories)):
        col_val = metric_categories[i]
        ax = axs_list[i]
        _legend = True if col_val == g_str or col_val == performance_str else False

        if (col_val == g_str or col_val == frobenius_str) and len(kernel_to_compare) > 1:
            def get_combined_latex_str(x):
                return "$d($"+ kernel_to_compare_to_latex_str(str(x["classical_kernel"])) + ", " + x["kernel_type"] + "$)$"
            melted_copy_ = melted_copy[melted_copy['metric_category'] == col_val]
            #create a new column which concatenates the string of classical_kernel and kernel_type 
            melted_copy_[combined_str] = melted_copy_.apply(lambda x: get_combined_latex_str(x), axis=1)
            sns.boxplot(
                data=melted_copy_,
                x="num_qubits",
                y="value",
                hue=combined_str,
                #hue_order=["FQK", "PQK", *kernel_to_compare],
                ax=ax,
                meanline=True,
                showmeans=False,
                boxprops={"linewidth": 2},
                medianprops=dict(color="black", alpha=1, linewidth=2, linestyle='-'),
                width=0.9,
                flierprops={"marker": "d", "markerfacecolor": "black"},
                legend=_legend,
                dodge=True, 
                palette=sns.color_palette(["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3"])
            )
        else:
            #Boxplot for the upper section
            sns.boxplot( 
                data=melted_copy[melted_copy['metric_category'] == col_val],
                x="num_qubits",
                y="value",
                hue=kernel_type_str,
                hue_order=hue_order,
                ax=ax,
                legend=_legend,
                boxprops={"linewidth": 2},
                medianprops=dict(color="black", alpha=1, linewidth=2, linestyle='-'),
                width=0.9,
                flierprops={"marker": "d", "markerfacecolor": "black"},
                palette=color_palette
            )
         
        # Set y-axis limits to break the axis between 1.5 and 4
        ax.set_ylabel(col_val)
        ax.set_xlabel(num_qubit_str)
        ax.set_xlim(-0.5, 8)
        ax.set_title("")
        low_y, high_y = ax.get_ylim()
        
        columnspacing=0.75
        fontsize = 17 #14

        ax.set_ylabel(col_val)

        if col_val == performance_str: 
            ax.set_ylim(0.45, high_y)
            #make text bold
            if dataset_name_str == "pennylane_hidden-manifold":
                dataset_name_str = "hidden-manifold"
            label_text = f"{dataset_name_str} \n ROC-AUC"
            ax.set_ylabel(f"{label_text}")
            ax.legend(title=None, loc = "lower center",  ncol=2, handlelength=1, handleheight=1, fontsize=fontsize, columnspacing=columnspacing)
        elif col_val == top_eigenvalue_str:
            #ax.set_ylim(0, 1.5), set ylog scale
            pass
        elif col_val == bandwidth_str:
            ax.set_yscale("log")
            ax.set_ylabel("bandwidth $c^*$")
            #get ylim
            #ax.set_ylim(low_y, 10**1)
            #ax.set_yticks([10**i for i in range(-3, 1, 1)])
        elif col_val == g_str:
            ax.set_yscale("log")
            ax.axhline(y=np.sqrt(N_train), color='r', linestyle='--')
            #ax.set_yticks([10**i for i in range(-1, 10, 2)])
            #fontsize = 10
            ax.legend(title=None, loc = "lower center",  ncol=1, handlelength=1, handleheight=1, fontsize=fontsize, columnspacing=columnspacing)

            ax.set_ylim(bottom=0.5*10**-2)
        elif col_val == frobenius_str:
            ax.set_yscale("log")
            #ax.set_yticks([10**i for i in range(-1, 10, 2)])
            ax.set_ylim(bottom=10**-4)

        elif col_val == varK_str:
            ax.set_yscale("log")
            ax.set_ylim(10**-7, 10**4)
        
        if show_abc:
            print("showing abc")
            ax.set_title(f"({chr(97+i).upper()})", weight='bold', size=20, color='black')

        # Hide spines and x-axis labels for a seamless break
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        for j, nq in enumerate(melted_copy["num_qubits"].unique()):
            ax.axvline(x=j-0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.75)
        
           
        if hide_xlabels:
            ax.set_xlabel("")
            ax.set_xticklabels([])        
            ax.set_xlabel("")
        else:
            ax.set_xlabel(num_qubit_str)


    #plt.tight_layout()
    if show_plt:
        plt.show()
    
    else:   
        if hide_xlabels:
            #write a text a letter for each column 
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}_xlabels_hidden_nolabel.png", dpi=500,  bbox_inches = 'tight')
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}_xlabels_hidden_nolabel.pdf", dpi=500,  bbox_inches = 'tight')
            
            
            #plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}_xlabels_hidden.png", dpi=500,  bbox_inches = 'tight')
            #plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}_xlabels_hidden.pdf", dpi=500,  bbox_inches = 'tight')
        else:
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}.png", dpi=500,  bbox_inches = 'tight')
            plt.savefig(f"{path_to_save}{dataset_name_str}_{encoding_circuit_name_str}_quantities_vs_n_cQ_and_cC_{kernel_to_compare}.pdf", dpi=500,  bbox_inches = 'tight')
            
