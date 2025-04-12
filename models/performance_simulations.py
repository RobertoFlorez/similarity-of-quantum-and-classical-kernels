
from data_tools.get_dataset import load_dataset
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, pairwise_kernels
from data_tools.tools import write_dic_results, merge_temporary_files, remove_temporary_files
from models.manual_kernels import K_PQK_with_different_gamma, variance_off_diagonal
from sklearn.gaussian_process.kernels import Matern
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from models.spectral_bias_and_target_aligment import eigendecomposition
from models.manual_kernels import *
from models.spectral_bias_and_target_aligment import get_spectral_bias_tool
import os


import numpy as np


Matern_Kernel_nu_1_5 = Matern(length_scale=1.0, nu=1.5)
Matern_Kernel_nu_2_5 = Matern(length_scale=1.0, nu=2.5)

method_mappping = {"Matern_Kernel_nu_1_5": Matern_Kernel_nu_1_5, "Matern_Kernel_nu_2_5": Matern_Kernel_nu_2_5}

kernel_cache = {}  # Dictionary to store precomputed kernels
def Get_K_train_K_test_from_params(params, X_train, X_test):
    """
    typically, params look like this:
    {'C': 0.0625, 'kernel': 'rbf', 'gamma': 0.5}
    """
    #create relevant params, copy of params without C and or alpha
    params_n = {}
    params_n["num_qubits"] = X_train.shape[1]
    for key, value in params.items():
        if key not in {"C", "alpha", "gamma"}:
            params_n[key] = value
        if params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params_n["gamma"] = params["gamma"]


    #copy the dictionary params_n
    kernel_key = params_n.copy()
    #remove the results_path from the dictionary
    kernel_key.pop("results_path")
    kernel_key = str(kernel_key)
    print("kernel_key", kernel_key)

    if params["kernel"] in method_mappping.keys(): #Only for Matern Kernel
        kernel_key = str(params_n)
        if kernel_key in kernel_cache:
            return kernel_cache[kernel_key]
        kernel = method_mappping[params["kernel"]]
        K_train, K_test = kernel(X_train, X_train), kernel(X_test, X_train)

        # Store the calculated kernel in the cache
        kernel_cache[kernel_key] = (K_train, K_test)
        return K_train, K_test

    
    if kernel_key in kernel_cache:
        print(kernel_key)
        print("kernel_key in kernel_cache")
        return kernel_cache[kernel_key]
    
    if params["kernel"] == "taylor_separable_rx":
        #if num_layers != 1 then we need to include this information where the 1 is currenlty located
        K_train = separable_rx_gram_matrix_fast(X_train, X_train, 1, polynomial_approximation=params["degree"], noise=False, simple_poly=False)
        K_test = separable_rx_gram_matrix_fast(X_test, X_train, 1, polynomial_approximation=params["degree"], noise=False, simple_poly=False)
    elif params["kernel"] == "rbf_poly":
        K_train = rbf_poly_approximation(X_train, X_train, gamma=params["gamma"], degree=params["degree"])
        K_test = rbf_poly_approximation(X_test, X_train, gamma=params["gamma"], degree=params["degree"])
    else:
        new_gamma = 1*params.get("gamma", None)*1# /params.get("degree", None)
        K_train = pairwise_kernels(X_train, metric=params['kernel'], degree=params.get("degree", None), gamma=new_gamma, filter_params=True)
        K_test = pairwise_kernels(X_test, X_train, metric=params['kernel'], degree=params.get("degree", None), gamma=new_gamma, filter_params=True)

    # Store the calculated kernel in the cache

    kernel_cache[kernel_key] = (K_train, K_test)
    return K_train, K_test

def wrapper_around_get_K_train_K_test_from_params_classical_performance(dictionary):
    X_train, X_test, y_train, y_test = load_dataset(dictionary["dataset_name"].decode(), dictionary["num_qubits"], dictionary["num_datapoints"][0], train_test_split_value = dictionary["num_datapoints"][1])
    K_train, K_test = Get_K_train_K_test_from_params(dictionary, X_train, X_test)   
    return K_train, K_test, X_train, X_test, y_train, y_test

C_to_save = 512.0
def LearnAndPredict_SVC(params, X_train, X_test, y_train, y_test, results_path, results_dic = {}, classical = False, classification_problem = True):
    """
    For a given set of hyperparameters, learn and predict using Support Vector Classifier and return accuracy and mse

    if precomputed_kernel = True, then X_train and X_test are precomputed kernels of shape (n_samples_train, n_samples_train) and (n_samples_test, n_samples_train) respectively
    """
    #del params["alpha"]
    if classification_problem == False:
        print("No classification problem, skipping SVC")
        return False

    start = time.time()
    params = params.copy()
    results_dic = results_dic.copy()

    if classical == True:
        kernel_method_str = params["kernel"]
        K_train, K_test = Get_K_train_K_test_from_params(params, X_train, X_test)
        kernel_time = time.time()
        #print("Kernel time", time.time()-start)
        if params["bandwidth"] >= 2:
            max_iter = 3*10**6
        else:
            max_iter = 6*10**5
            
        del params["bandwidth"] #This has note been check for quantum kernels
        del params["kernel"] #Because we use this information to calculate the kernel, but we don't need it in params
        
        params_to_use = params.copy()
        for key in params.keys(): #delete everything except C and gamma
            if key not in ["C", "gamma"]:
                del params_to_use[key]

        svm_classifier = SVC(**params_to_use, kernel = "precomputed", cache_size = 500, tol=10**-3, max_iter = max_iter, random_state=0)
        try:
            svm_classifier.fit(K_train, y_train)
            print("number of iterations", svm_classifier.n_iter_)
            y_pred_svm_train = svm_classifier.predict(K_train)
            y_pred_svm = svm_classifier.predict(K_test)

        except Exception as e:
            print("Error in fitting SVC", e)
            y_pred_svm = np.zeros(y_test.shape)
            
        #print("Classification time", time.time()-start)


    else:
        del params["gamma"] #gamma is not used when kernel is precomputed
        K_train = X_train #precomputed kernel of shape (n_samples_train, n_samples_train)
        K_test = X_test #precomputed kernel of shape (n_samples_test, n_samples_train)
        del params["bandwidth"] #This has note been check for quantum kernels
        del params["seed"] #Because we use this information to calculate the kernel, but we don't need it in params


        #results_dic["K_train"] = K_train
        #results_dic["K_test"] = K_test
        svm_classifier = SVC(**params, kernel = "precomputed", random_state=0) #max_iter = 10**7)
        svm_classifier.fit(K_train, y_train)
        y_pred_svm_train = svm_classifier.predict(K_train)
        y_pred_svm = svm_classifier.predict(K_test)
        print("Classification time", time.time()-start)

    #check if the number of unique classes is the same in y_pred_svm and y_test
    if len(np.unique(y_pred_svm)) != len(np.unique(y_test)):
        print("very bad prediction!, saving scores as nan")
        y_pred_svm = np.zeros(y_test.shape)
        y_pred_svm_train = np.zeros(y_train.shape)

        accuracy_svm = np.nan
        mse_svm = np.nan
        roc_auc_score_svm = np.nan
        precision_svm = np.nan
        recall_svm = np.nan
        f1_score_svm = np.nan
        params["num_iterations_svm"] = np.nan
        params["var_alpha"] = np.nan

        
    else:
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        mse_svm = mean_squared_error(y_test, y_pred_svm)    

        roc_auc_score_svm = roc_auc_score(y_test, y_pred_svm)
        print(precision_score(y_test, y_pred_svm))
        precision_svm = precision_score(y_test, y_pred_svm)
        recall_svm = recall_score(y_test, y_pred_svm)
        f1_score_svm = f1_score(y_test, y_pred_svm)

        params["num_iterations_svm"] = svm_classifier.n_iter_
        params["var_alpha"] = np.var(svm_classifier.dual_coef_.flatten())

    
    ignored_keys = ["X_train", "X_test", "y_train", "y_test", "K_test", "density_matrices_train", "density_matrices_test", "eigenvectors"]

    if classical == True:
        params["kernel"] = kernel_method_str     #This is for Matern Kernel, to write the string in the results_dic

        eigenvalue_spectrum, eigenvectors, cumul, _ = get_spectral_bias_tool(K_train, X_train, y_train)
        params["eigenvalues"] = eigenvalue_spectrum
        params["ck"] = cumul
        params["K_max_eig"] = np.max(np.real(params["eigenvalues"]))
        params["varK_train"] = variance_off_diagonal(K_train)
        params["varK_train_with_diagonal"] = np.var(K_train.flatten()) 

        if params["C"] == C_to_save: #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
            results_dic["K_train"] = K_train
        else:
            ignored_keys.append("K_train")
        
    else:
        ignored_keys.append("K_train")

        params["varK_train"] = results_dic["varK_train"]
        params["varK_train_with_diagonal"] = np.var(K_train.flatten())
        params["K_max_eig"] = np.max(results_dic["eigenvalues"])
    params["rankK"] = np.linalg.matrix_rank(K_train)


    params["K_mean_train"] = np.mean(K_train.flatten())
    print("K_mean_train", params["K_mean_train"])
    params["K_min_train"] = np.min(K_train.flatten())
    params["d"] = np.linalg.matrix_rank(K_train)


    params["y_pred"] = y_pred_svm
    params["y_pred_train"] = y_pred_svm_train


    results_dic["accuracy"] = accuracy_svm
    results_dic["mse"] = mse_svm
    results_dic["roc_auc_score"] = roc_auc_score_svm

    #include accuarcym, mse, roc_auc_score, of train set
    results_dic["accuracy_train"] = accuracy_score(y_train, y_pred_svm_train)
    results_dic["mse_train"] = mean_squared_error(y_train, y_pred_svm_train)
    results_dic["roc_auc_score_train"] = roc_auc_score(y_train, y_pred_svm_train)


    results_dic["precision"] = precision_svm
    results_dic["recall"] = recall_svm
    results_dic["f1_score"] = f1_score_svm
    results_dic["regression_method"] = "SVC"

    #add params to results_dic
    for key, value in params.items():
        results_dic[key] = value
    for key in ignored_keys:
        if key in results_dic.keys():
            del results_dic[key]
    end_calculation = time.time()
    
    calculation_time = end_calculation - start
    results_dic["calculation_time"] = calculation_time
    print("roc_auc_score", roc_auc_score_svm)

    #print("results!!!", results_dic)
    return results_dic



def LearnAndPredict_SVR(params, X_train, X_test, y_train, y_test, results_path, results_dic = {}, classical = False, classification_problem = True):
    """
    For a given set of hyperparameters, learn and predict using Support Vector Regressor and return accuracy and mse
    if precomputed_kernel = True, then X_train and X_test are precomputed kernels of shape (n_samples_train, n_samples_train) and (n_samples_test, n_samples_train) respectively

    """
    #del params["alpha"]
    start = time.time()

    params = params.copy()
    results_dic = results_dic.copy()

    if classical == True:
        kernel_method_str = params["kernel"]
        K_train, K_test = Get_K_train_K_test_from_params(params, X_train, X_test)
        del params["kernel"] #Because we use this information to calculate the kernel, but we don't need it in params
        svr_classifier = SVR(**params, kernel = "precomputed")
        svr_classifier.fit(K_train, y_train)

        y_pred_svr = svr_classifier.predict(K_test)
    else:
        del params["gamma"] #gamma is not used when kernel is precomputed
        K_train = X_train #precomputed kernel of shape (n_samples_train, n_samples_train)
        K_test = X_test #precomputed kernel of shape (n_samples_test, n_samples_train)
        svr_classifier = SVR(**params, kernel = "precomputed")
        svr_classifier.fit(K_train, y_train)
        y_pred_svr = svr_classifier.predict(K_test)
    
    if classification_problem:
        y_pred_svr = np.where(y_pred_svr > 0.5, 1, 0)
        accuracy_svr = accuracy_score(y_test, y_pred_svr)
    else:
        accuracy_svr = r2_score(y_test, y_pred_svr)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    #params = AddMetadata(params, "blobs", "SVR", X_train.shape[0], params["gamma"], accuracy_svr, "local", mse_svr)
    

    results_dic["accuracy"] = accuracy_svr
    results_dic["mse"] = mse_svr
    results_dic["regression_method"] = "SVR"

    #This is for Matern Kernel, to write the string in the results_dic
    
    if classical == True:
        params["kernel"] = kernel_method_str
    
    ignored_keys = ["X_train", "X_test", "y_train", "y_test", "K_train", "K_test", "density_matrices_train", "density_matrices_test", "ck", "eigenvalues", "eigenvectors"]
    #add params to results_dic
    for key, value in params.items():
        results_dic[key] = value
    for key in ignored_keys:
        if key in results_dic.keys():
            del results_dic[key]

    end = time.time()
    calculation_time = end - start
    results_dic["calculation_time"] = calculation_time
    return results_dic


def LearnAndPredict_KRR(params, X_train, X_test, y_train, y_test, results_path, results_dic = {}, classical = False, classification_problem = True):
    """
    For a given set of hyperparameters, learn and predict using Kernel Ridge Regressor and return accuracy and mse
    """
    start = time.time()

    params = params.copy()
    results_dic = results_dic.copy()

    
    if classical == True:
        kernel_method_str = params["kernel"]
        K_train, K_test = Get_K_train_K_test_from_params(params, X_train, X_test)
        kernel_time = time.time()
        #print("Kernel time", time.time()-start)
                    
        params_to_use = params.copy()
        for key in params.keys(): #delete everything except C and gamma
            if key not in ["alpha", "gamma"]:
                del params_to_use[key]

        krr_classifier = KernelRidge(**params_to_use, kernel = "precomputed")
        krr_classifier.fit(K_train, y_train)
        y_pred_krr = krr_classifier.predict(K_test)
        
    else:
        del params["gamma"] #gamma is not used when kernel is precomputed
        K_train = X_train
        K_test = X_test
        #results_dic["K_train"] = K_train
        #results_dic["K_test"] = K_test
        krr_classifier = KernelRidge(**params, kernel = "precomputed")
        krr_classifier.fit(K_train, y_train)
        y_pred_krr = krr_classifier.predict(K_test)

    if classification_problem:
        y_pred_krr = np.where(y_pred_krr > 0.5, 1, 0)
        accuracy_krr = accuracy_score(y_test, y_pred_krr)
    else:
        accuracy_krr = r2_score(y_test, y_pred_krr)

    mse_krr = mean_squared_error(y_test, y_pred_krr)
    #params = AddMetadata(params, "blobs", "KRR", X_train.shape[0], params["gamma"], accuracy_krr, "local", mse_krr)
    
    results_dic["accuracy"] = accuracy_krr
    results_dic["mse"] = mse_krr
    results_dic["regression_method"] = "KRR"
    
    ignored_keys = ["X_train", "X_test", "y_train", "y_test", "K_test", "density_matrices_train", "density_matrices_test", "eigenvectors"]


    if classical == True:
        params["kernel"] = kernel_method_str     #This is for Matern Kernel, to write the string in the results_dic

        eigenvalue_spectrum, eigenvectors, cumul, _ = get_spectral_bias_tool(K_train, X_train, y_train)
        params["eigenvalues"] = eigenvalue_spectrum
        params["ck"] = cumul
        params["K_max_eig"] = np.max(np.real(params["eigenvalues"]))
        params["varK_train"] = variance_off_diagonal(K_train)
        params["varK_train_with_diagonal"] = np.var(K_train.flatten()) 

        if params["alpha"] == 1/(2*C_to_save): #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
            results_dic["K_train"] = K_train
        else:
            ignored_keys.append("K_train")
    else:
        ignored_keys.append("K_train")

        params["varK_train"] = results_dic["varK_train"]
        params["varK_train_with_diagonal"] = np.var(K_train.flatten())
        params["K_max_eig"] = np.max(results_dic["eigenvalues"])
    params["rankK"] = np.linalg.matrix_rank(K_train)

    params["K_mean_train"] = np.mean(K_train.flatten())
    print("K_mean_train", params["K_mean_train"])
    params["K_min_train"] = np.min(K_train.flatten())
    params["d"] = np.linalg.matrix_rank(K_train)


    params["y_pred"] = y_pred_krr
    params["y_pred_train"] = krr_classifier.predict(K_train)


    #include accuarcym, mse, roc_auc_score, of train set
    results_dic["mse_train"] = mean_squared_error(y_train, params["y_pred_train"])

    #add params to results_dic
    for key, value in params.items():
        results_dic[key] = value
    for key in ignored_keys:
        if key in results_dic.keys():
            del results_dic[key]
    
    end = time.time()

    calculation_time = end - start
    results_dic["calculation_time"] = calculation_time
    return results_dic


def get_poly_combinations(param_grid_sv, param_grid_kr, gamma_list):
    param_grid_poly_sv = param_grid_sv.copy()
    param_grid_poly_kr = param_grid_kr.copy()
    param_grid_poly_sv["kernel"] = ['poly']
    param_grid_poly_kr["kernel"] = ['poly']
    param_grid_poly_sv["degree"] = [2, 3, 4]
    param_grid_poly_kr["degree"] = [2, 3, 4]
    param_grid_poly_sv["gamma"] = gamma_list
    param_grid_poly_kr["gamma"] = gamma_list

    return param_grid_poly_sv, param_grid_poly_kr

def get_linear_combinations(param_grid_sv, param_grid_kr):
    param_grid_linear_sv = param_grid_sv.copy()
    param_grid_linear_kr = param_grid_kr.copy()
    param_grid_linear_sv["kernel"] = ['linear']
    param_grid_linear_kr["kernel"] = ['linear']

    return param_grid_linear_sv, param_grid_linear_kr

def get_rbf_and_sigmoid_combinations(param_grid_sv, param_grid_kr, gamma_list, kernels_ = ["rbf", "sigmoid"]):
    param_grid_sv["kernel"] = kernels_
    param_grid_kr["kernel"] = kernels_
    param_grid_sv["gamma"] = gamma_list
    param_grid_kr["gamma"] = gamma_list

    return param_grid_sv, param_grid_kr
def get_params_combinations(classical, C_list, gamma_list, full_grid, num_qubits):

     # Create a parameter grid
    param_grid_sv = {
        'C': C_list,  # Penalty parameter C for SVC/SVR
    }
    param_grid_kr = {
        'alpha': 1/(2*C_list), # regularization parameter for KRR
    }

    param_grid_matern = {
    'C': C_list,  # Penalty parameter C for SVC/SVR
    "kernel": ["Matern_Kernel_nu_2_5", "Matern_Kernel_nu_1_5" ]
    }

    param_combinations = list()

    if classical == False:
        param_grid_kr["gamma"] = gamma_list
        param_grid_sv["gamma"] = gamma_list
        param_combinations += list(ParameterGrid(param_grid_sv))
        param_combinations += list(ParameterGrid(param_grid_kr))    
    elif classical == True:
        if full_grid == "Full":
            ## Add polynomial kernel
            param_grid_poly_sv, param_grid_poly_kr = get_poly_combinations(param_grid_sv, param_grid_kr, gamma_list)
            param_combinations += list(ParameterGrid(param_grid_poly_sv))
            param_combinations += list(ParameterGrid(param_grid_poly_kr))

            ## add linear
            param_grid_linear_sv, param_grid_linear_kr = get_linear_combinations(param_grid_sv, param_grid_kr)
            param_combinations += list(ParameterGrid(param_grid_linear_sv))
            param_combinations += list(ParameterGrid(param_grid_linear_kr))

            ## Add sigmoid, rbf kernel
            param_grid_rbf_and_sigmoid_sv, param_grid_rbf_and_sigmoid_kr = get_rbf_and_sigmoid_combinations(param_grid_sv, param_grid_kr, gamma_list)
            param_combinations += list(ParameterGrid(param_grid_rbf_and_sigmoid_sv))
            param_combinations += list(ParameterGrid(param_grid_rbf_and_sigmoid_kr))

            param_combinations += list(ParameterGrid(param_grid_matern))
        elif full_grid =="PolySV":
            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['poly']
            param_grid_poly_sv["degree"] = [2] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1] #/(num_qubits)]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))
        elif full_grid == "taylor_separable_rx":
            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['taylor_separable_rx']
            param_grid_poly_sv["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))

        elif full_grid == "RBFSV":
            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))

        elif full_grid == "RBFandPolySV":
            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))

            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['poly']
            param_grid_poly_sv["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))
        elif full_grid == "RBFandRBFPolySV":
            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['rbf_poly']
            param_grid_poly_sv["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))

            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))
        elif full_grid == "RBFandPolyKRR":
            param_grid_RBF = param_grid_kr.copy()
            param_grid_RBF["kernel"] = ['rbf']
            param_grid_RBF["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF))

            param_grid_poly = param_grid_kr.copy()
            param_grid_poly["kernel"] = ['poly']
            param_grid_poly["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly))

        elif full_grid == "RBFandPolySV_gamma_HamEvol":
            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [0.1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))

            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['poly']
            param_grid_poly_sv["degree"] = [4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [0.1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))

        elif full_grid == "RBFandPolySV_taylor_separable_rx":
            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))

            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['poly']
            param_grid_poly_sv["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))

            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['taylor_separable_rx']
            param_grid_poly_sv["degree"] = [1,2,3,4] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))
        elif full_grid == "Partial":
            #add linear
            param_grid_linear_sv, param_grid_linear_kr = get_linear_combinations(param_grid_sv, param_grid_kr)
            param_combinations += list(ParameterGrid(param_grid_linear_sv))
            param_combinations += list(ParameterGrid(param_grid_linear_kr))

            ## Add sigmoid, rbf kernel
            param_grid_rbf_and_sigmoid_sv, param_grid_rbf_and_sigmoid_kr = get_rbf_and_sigmoid_combinations(param_grid_sv, param_grid_kr, gamma_list)
            param_combinations += list(ParameterGrid(param_grid_rbf_and_sigmoid_sv))
            param_combinations += list(ParameterGrid(param_grid_rbf_and_sigmoid_kr))

            param_combinations += list(ParameterGrid(param_grid_matern))
        
        elif full_grid == "simple":
            param_grid_RBF_sv = param_grid_sv.copy()
            param_grid_RBF_sv["kernel"] = ['rbf']
            param_grid_RBF_sv["gamma"] = [1]
            param_combinations += list(ParameterGrid(param_grid_RBF_sv))

            param_grid_poly_sv = param_grid_sv.copy()
            param_grid_poly_sv["kernel"] = ['poly']
            param_grid_poly_sv["degree"] = [3] #[i for i in range(0, num_qubits)] 
            param_grid_poly_sv["gamma"] = [1]#[1/(num_qubits), 1/(num_qubits*2)] 
            param_combinations += list(ParameterGrid(param_grid_poly_sv))
            
        elif full_grid == "Small":
            ## Add rbf kernel
            param_grid_rbf_sv, param_grid_rbf_kr = get_rbf_and_sigmoid_combinations(param_grid_sv, param_grid_kr, gamma_list, kernels_ = ["rbf"])
            param_combinations += list(ParameterGrid(param_grid_rbf_sv))
            param_combinations += list(ParameterGrid(param_grid_rbf_kr))
        
    return param_combinations

def get_grid_list(full_grid, X_train, num_qubits):
    if full_grid == "Full":
        C_list = np.array([0.5, 32,64,128, 512, 1024*2, 1024*3])
        gamma_list = np.array([0.1, 0.25, 0.5, 1,2,3,4, 5, X_train.shape[1]*X_train.var()])*1/(X_train.shape[1]*X_train.var())
    elif full_grid == "Partial":
        C_list = np.array([32.0, 512.0, 1024.0*2])
        gamma_list = [1]
    elif full_grid == "Small":
        C_list = np.array([0.5]) #10**-6, 10**-5, 10**-4
        gamma_list = np.array([1, 0.5])*1/(X_train.shape[1]*X_train.var())
    elif full_grid == "PolySV" or full_grid=="RBFSV":
        #C_list = np.array([0.5, 32,64,128, 512, 1024*2, 1024*3])
        C_list = np.array([32.0, 64.0, 128.0, 512.0, 1024.0])
        gamma_list = [1]  # 1/(num_qubits*2) for 4
    elif full_grid == "RBFandPolySV" or "simple" or "RBFandPolySV_taylor_separable_rx" or "taylor_separable_rx" or "RBFSV" or full_grid == "RBFandPolyKRR":
        C_list = np.array([32.0, 64.0, 128.0, 512.0, 1024.0])
        gamma_list = [1]
    elif full_grid == "RBFandPolySV_gamma_HamEvol":
        C_list = np.array([512.0])
        gamma_list = [10**-1]
    elif full_grid == "PolySV_gamma_HamEvol":
        C_list = np.array([512.0])
        gamma_list = [10]
    return C_list, gamma_list

def get_performance_for_dic(results_dic, results_path, classical, include_svc = True, include_svr = False, 
                            include_krr = False, full_grid = False, classification_problem =True):
    """
    For a given set of experiment parameters, learn and predict using SVC, SVR and KRR and return accuracy and mse. 
    If classical = True, then SVC, SVR and KRR are calculated using classical kernels. 
    If classical = False, then SVC, SVR and KRR are calculated using quantum kernels and must be precomputed and stored in results_dic["K_train"] and results_dic["K_test"]

    The following parameters must be in results_dic:
    - dataset_name
    - num_datapoints
    - num_qubits
    - bandwidth
    - K_train (if classical = False)
    - K_test (if classical = False)
    """
    start_time = time.time()
    print("classical", classical, "include_svc", include_svc, "include_svr", include_svr, 
          "include_krr", include_krr, "full_grid", full_grid, "classification_problem", classification_problem)
    results_dic = results_dic.copy()
    dataset_name = results_dic["dataset_name"]
    num_datapoints = results_dic["num_datapoints"]
    train_test_split_value = results_dic["train_test_split_value"]
    seed = results_dic["seed"]
    num_qubits = results_dic["num_qubits"]
    bandwidth = results_dic["bandwidth"]

    if type(dataset_name) == bytes:
        dataset_name = dataset_name.decode()
    if dataset_name.startswith("quantum_"):
        classification_problem = False
    
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, num_qubits, num_datapoints, train_test_split_value = train_test_split_value, seed = seed)
    C_list, gamma_list = get_grid_list(full_grid, X_train, num_qubits)


    X_train, X_test, y_train, y_test = bandwidth* X_train, bandwidth* X_test, y_train, y_test
    
    
    if classical == True:
        param_combinations = get_params_combinations(classical, C_list, gamma_list, full_grid, num_qubits)
        results_dic["method"] = "classical"
        K_train = X_train # if classical, then K_train and K_test are not used and X_train and X_test are used instead
        K_test = X_test
    elif classical == False:
        param_combinations = get_params_combinations(classical, C_list, gamma_list, full_grid, None)
        print(param_combinations)
        if results_dic["method"] == "FQK".encode() or results_dic["method"] == "FQK":
            K_train = results_dic["K_train"] #if quantum, then K_train and K_test are precomputed kernels
            K_test = results_dic["K_test"]
    else:
        raise ValueError("method not recognized")
    
    result_dic_list = []
    print("Number of parameter combinations:", len(param_combinations))  
    for idx, params_o in enumerate(param_combinations):
        if idx % 1 == 0:
            print(f"{idx}/{len(param_combinations)}") 
            #print("params_o:", params_o)
        if results_dic["method"] == "PQK": 
            K_train = K_PQK_with_different_gamma(results_dic["K_train"], gamma_original = results_dic["gamma_original"], gamma_new = params_o["gamma"] )
            K_test = K_PQK_with_different_gamma(results_dic["K_test"], gamma_original = results_dic["gamma_original"], gamma_new = params_o["gamma"])
            results_dic["gamma"] = params_o["gamma"]
            
        if "C" in params_o.keys():
            if include_svc:
                params_o["bandwidth"] = bandwidth 
                params_o["seed"] = seed

                results_dic_out = LearnAndPredict_SVC(params_o, K_train, K_test, y_train, y_test, 
                                                                                        results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)
                if params_o["C"] == C_to_save: #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
                    write_dic_results(results_path + f"{idx}.h5", results_dic_out)
                    #print(results_dic_out)
                    results_dic_out.pop("K_train", None)

                if results_dic_out == False:
                    continue
                else:
                    result_dic_list.append(pd.DataFrame([results_dic_out]))

            if include_svr:
                results_dic_out = LearnAndPredict_SVR(params_o, K_train, K_test, y_train, y_test, 
                                                                                        results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)    
                result_dic_list.append(pd.DataFrame([results_dic_out]))

        if "alpha" in params_o.keys():
            #Matern not implemented for KRR yet
            if include_krr:
                results_dic_out = LearnAndPredict_KRR(params_o, K_train, K_test, y_train, y_test, 
                                                                                        results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)
                
                if params_o["alpha"] == 1/(2*C_to_save): #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
                    write_dic_results(results_path + f"{idx}.h5", results_dic_out)
                    #print(results_dic_out)
                    results_dic_out.pop("K_train", None)
                result_dic_list.append(pd.DataFrame([results_dic_out]))
        
    df = pd.concat(result_dic_list, axis=0, sort=False, ignore_index=True)
    end_time = time.time()  
    print("T_all_calc:", end_time - start_time)
    pickle_time = time.time()
    df.reset_index(drop=True, inplace=True)
    df.to_feather(results_path + ".feather")
    feather_time = time.time()
    print("T_write .feather:", feather_time  - pickle_time)
    root_folder = os.path.dirname(results_path) 
    #print("root_folder", root_folder)
    


def get_performance_for_dic_classical_hyperparameters(results_dic, results_path, classical, include_svc = True, include_svr = False, 
                            include_krr = False, full_grid = False, classification_problem =True):
    """
    For a given set of experiment parameters, learn and predict using SVC, SVR and KRR and return accuracy and mse. 
    If classical = True, then SVC, SVR and KRR are calculated using classical kernels. 
    If classical = False, then SVC, SVR and KRR are calculated using quantum kernels and must be precomputed and stored in results_dic["K_train"] and results_dic["K_test"]

    The following parameters must be in results_dic:
    - dataset_name
    - num_datapoints
    - num_qubits
    - bandwidth
    - K_train (if classical = False)
    - K_test (if classical = False)
    """
    start_time = time.time()
    results_dic = results_dic.copy()
    dataset_name = results_dic["dataset_name"]
    num_datapoints = results_dic["num_datapoints"]
    train_test_split_value = results_dic["train_test_split_value"]
    num_qubits = results_dic["num_qubits"]
    bandwidth = results_dic["bandwidth"]
    seed = results_dic["seed"]
    
    if type(dataset_name) == bytes:
        dataset_name = dataset_name.decode()
    if dataset_name.startswith("quantum_"):
        classification_problem = False
    
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, num_qubits, num_datapoints, train_test_split_value = train_test_split_value, seed = seed)
    X_train, X_test, y_train, y_test = bandwidth* X_train, bandwidth* X_test, y_train, y_test
    
    
    if classical == True:
        results_dic["method"] = "classical"
        K_train = X_train # if classical, then K_train and K_test are not used and X_train and X_test are used instead
        K_test = X_test
    elif classical == False:
        if results_dic["method"] == "FQK".encode() or results_dic["method"] == "FQK":
            K_train = results_dic["K_train"] #if quantum, then K_train and K_test are precomputed kernels
            K_test = results_dic["K_test"]
    else:
        raise ValueError("method not recognized")
    
    result_dic_list = []

    params_o = results_dic.copy()
    
    if "C" in params_o.keys():
        if include_svc:
            params_o["bandwidth"] = bandwidth 
            results_dic_out = LearnAndPredict_SVC(params_o, K_train, K_test, y_train, y_test, 
                                                                                    results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)
            print("svc is done")
            if params_o["C"] == C_to_save: #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
                write_dic_results(results_path + f".h5", results_dic_out)
                #print(results_dic_out)
                results_dic_out.pop("K_train", None)

            if results_dic_out == False:
                pass
            else:
                result_dic_list.append(pd.DataFrame([results_dic_out]))

        if include_svr:
            results_dic_out = LearnAndPredict_SVR(params_o, K_train, K_test, y_train, y_test, 
                                                                                    results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)    
            result_dic_list.append(pd.DataFrame([results_dic_out]))
    print("printing params", params_o)
    if "alpha" in params_o.keys():
            #Matern not implemented for KRR yet
        if include_krr:
            results_dic_out = LearnAndPredict_KRR(params_o, K_train, K_test, y_train, y_test, 
                                                                                    results_path, results_dic=results_dic, classical=classical, classification_problem = classification_problem)
            
            if params_o["alpha"] == 1/(2*C_to_save): #Quick and dirty solution to saving the kernel matrices only once for the classical kernels
                write_dic_results(results_path + f".h5", results_dic_out)
                #print(results_dic_out)
                results_dic_out.pop("K_train", None)
            result_dic_list.append(pd.DataFrame([results_dic_out]))
        
    df = pd.concat(result_dic_list, axis=0, sort=False, ignore_index=True)
    end_time = time.time()  

    print("T_all_calc:", end_time - start_time)
    pickle_time = time.time()
    df.reset_index(drop=True, inplace=True)
    df.to_feather(results_path + ".feather")
    feather_time = time.time()
    print("T_write .feather:", feather_time  - pickle_time)
    root_folder = os.path.dirname(results_path) 

def get_performance_for_dic_classical_hyperparameters_wrapper(experiment_dic, results_path, constants):
    return get_performance_for_dic_classical_hyperparameters(experiment_dic, results_path, *constants)
    
def get_performance_for_dic_wrapper(experiment_dic, results_path, constants):
    print(results_path)
    return get_performance_for_dic(experiment_dic, results_path, *constants)