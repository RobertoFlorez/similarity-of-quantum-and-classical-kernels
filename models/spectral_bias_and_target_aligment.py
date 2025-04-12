import numpy as np	
import scipy as sp


def eigendecomposition(K):
    """Returns ordered eigenvalues and normalized eigenvectors of a kernel matrix."""
    eigenvalue_spectrum, eigenvectors = np.linalg.eigh(1/K.shape[0] * K)
    inds_sort = np.argsort(eigenvalue_spectrum)
    inds_sort = inds_sort[::-1]
    eigenvalue_spectrum = eigenvalue_spectrum[inds_sort]
    eigenvectors = eigenvectors[:,inds_sort]
    return eigenvalue_spectrum, eigenvectors

def get_spectral_bias_tool(K_train_for_analytical, X_train_for_analytical, y_train_for_analytical, P_list = None, lamb = None, sigma = 0):
    """
    Calculates:
    - eigenvalue and eigenvector spectrum
    - Cumulative Power #y_train must be a column vector (num_samples, dim_of_each_y)
    - Analytical Eg

    if lamb is None or P_list is None, then it does not calculate Eg. 

    returns eigenvalue_spectrum, eigenvectors, cumul, theory_lc
    """
    #Spectra
    eigenvalue_spectrum, eigenvectors = eigendecomposition(K_train_for_analytical)    
    #Cumulative Power
    #print("y_train_for_analytical.shape", y_train_for_analytical.shape)
    #print("eigenvectors.shape", eigenvectors.shape)
    #if y is a 1d row vector, transform it to a column vector
    if len(y_train_for_analytical.shape) == 1:
        y_train_for_analytical = y_train_for_analytical.reshape(-1, 1)
        #print("y_train_for_analytical reshaped.shape", y_train_for_analytical.shape)
    power = np.sum((eigenvectors.T @ y_train_for_analytical)**2, axis = 1)
    cumul = np.cumsum(power) / np.sum(power)

    if lamb is not None and P_list is not None:
        #Analytical Eg
        teacher = np.mean((eigenvectors.T @ y_train_for_analytical)**2, axis = 1)
        theory_lc = None
    else:
        theory_lc = None

    return eigenvalue_spectrum, eigenvectors, cumul, theory_lc