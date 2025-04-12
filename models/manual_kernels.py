from qiskit.quantum_info import entropy, Statevector, DensityMatrix
import pennylane as qml
from squlearn.util import Executor
import numpy as np
from functools import reduce
from scipy.special import factorial
import time
from scipy.interpolate import approximate_taylor_polynomial
from sklearn.metrics.pairwise import polynomial_kernel


factorial_list_upto_10 = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
# Pauli X
X = np.array([[0,1],[1,0]])
# Pauli Y
Y = np.array([[0,-1j],[1j,0]])
# Pauli Z
Z = np.array([[1,0],[0,-1]])

pauli = [X, Y, Z]


def identity_wrap(operator, index, num_qubits):
    """
    Wraps an element in tensor identity products

    Parameters
    -----------
    N: integer
        2^N size of Hilbert space
    index: integer 
        position in tensor product, ie:  I x I x I x A[index] x I
    operator: 
        desired element to wrap in, ie: A[index] in the above example 
    """
    N = num_qubits
    # Create a list of identity matrices
    identities = [np.eye(2) for _ in range(N)]

    # Replace the matrix at the specified index with the given operator
    identities[index] = operator

    # Compute the tensor product using functools.reduce
    result = reduce(np.kron, identities)
    
    return result

def projected_rbf(rho_1, rho_2,gamma, reduced_operator = None):
    """
    Computes the projected RBF Projected kernel between two density matrices

    Parameters
    -----------
    rho_1: numpy array
        density matrix 1
    rho_2: numpy array
        density matrix 2
    gamma: float

    reduced_operator: boolean, if true calculated using reduced operator P_i and full rho, else calculate reduced density matrices. Default is False
    Both are equally fast
    
    """
    n_qubits = int(np.log2(rho_1.shape[0]))
    rho_reduced_1 = np.zeros((n_qubits, 2, 2), dtype=np.complex128)
    rho_reduced_2 = np.zeros((n_qubits, 2, 2), dtype=np.complex128)
    A = 0
    if reduced_operator is True:
        for i in range(n_qubits):
            for P in pauli:
                P_i = identity_wrap(n_qubits, i, P)
                A += (np.trace(P_i @ rho_1) - np.trace(P_i @ rho_2))**2
    else:
        for i in range(n_qubits):
            rho_reduced_1[i] = qml.math.reduce_dm(rho_1, [i])
            rho_reduced_2[i] = qml.math.reduce_dm(rho_2, [i])
            for P in pauli:
                A += (np.trace(P @ rho_reduced_1[i]) - np.trace(P @ rho_reduced_2[i]))**2
    return np.exp(-gamma * A)

def get_rho_matrices(fmap, X_distribution, n_qubits):
    rho_matrices = np.zeros((len(X_distribution), 2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for i in range(len(X_distribution)):
        rho_matrices[i] = DensityMatrix(fmap.get_circuit(X_distribution[i], None)).data
    return rho_matrices

    
   

def K_FQK_and_PQK(fmap, X_distribution, gamma, n_qubits, X_distribution_test = None, reduced_local_operator = True, rho_matrices_train = None):
    """

    Parameters
    -----------
    fmap: FeatureMap object
    X_distribution: numpy array
        training data
    gamma: float  Kij = exp(-gamma * A), A = sum_Pk (tr(P_k @ rho_1i) - tr(P_k @ rho_2j))^2
    n_qubits: int
    X_distribution_test: numpy array
        test data
    reduced_operator: boolean, if true calculated using reduced operator P_i and full rho, else calculate reduced density matrices. Default is False
    Both are equally fast
    """
    if X_distribution_test is None: #K(X_distribution, X_distribution)
        if rho_matrices_train is None:
            rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)

        K_manual_FQK = np.zeros((len(X_distribution), len(X_distribution)))
        K_manual_PQK = np.zeros((len(X_distribution), len(X_distribution)))
        for i in range(len(X_distribution)):
            for j in range(len(X_distribution)):
                K_manual_PQK[i, j] = np.real(projected_rbf(rho_matrices_train[i], rho_matrices_train[j], gamma, reduced_local_operator))
                K_manual_FQK[i, j] = np.real(np.trace(rho_matrices_train[i] @ rho_matrices_train[j]))
        rho_matrices = [rho_matrices_train]
                
    else: #K(X_distribution, X_distribution_test)
        if rho_matrices_train is None:
            rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)
    
        rho_matrices_test = get_rho_matrices(fmap, X_distribution_test, n_qubits)
        
        K_manual_FQK = np.zeros((len(X_distribution), len(X_distribution_test)))
        K_manual_PQK = np.zeros((len(X_distribution), len(X_distribution_test)))
        

        for i in range(len(X_distribution)):
            for j in range(len(X_distribution_test)):
                K_manual_PQK[i, j] = np.real(projected_rbf(rho_matrices_train[i], rho_matrices_test[j], gamma, reduced_local_operator))
                K_manual_FQK[i, j] = np.real(np.trace(rho_matrices_train[i] @ rho_matrices_test[j]))
        rho_matrices = [rho_matrices_train, rho_matrices_test]
    return K_manual_FQK, K_manual_PQK, rho_matrices

def K_PQK(fmap, X_distribution, gamma, n_qubits, X_distribution_test = None, reduced_local_operator = None):
    """Calculates the projected RBF kernel between two distributions of data points, using the manual implementation
    Parameters
    -----------
    fmap: 
        feature map
    X_distribution: numpy array
        distribution of data points
    gamma: float
    n_qubits: int
        number of qubits
    X_distribution_test: numpy array
        distribution of data points for testing
    reduced_operator: boolean, if true calculated using reduced operator P_i and full rho, else calculate reduced density matrices. Default is False
    """
    if X_distribution_test is None: #K(X_distribution, X_distribution)
        rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)
        K_manual_PQK = np.zeros((len(X_distribution), len(X_distribution)))
        for i in range(len(X_distribution)):
            for j in range(len(X_distribution)):
                K_manual_PQK[i, j] = np.real(projected_rbf(rho_matrices_train[i], rho_matrices_train[j], gamma, reduced_local_operator))
        rho_matrices = [rho_matrices_train]



    else: # K(X_distribution, X_distribution_test)
        rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)
        rho_matrices_test = get_rho_matrices(fmap, X_distribution_test, n_qubits)


        K_manual_PQK = np.zeros((len(X_distribution), len(X_distribution_test)))
        for i in range(len(X_distribution)):
            for j in range(len(X_distribution_test)):
                K_manual_PQK[i, j] = np.real(projected_rbf(rho_matrices_train[i], rho_matrices_test[j], gamma, reduced_local_operator))
        rho_matrices = [rho_matrices_train, rho_matrices_test]

    return K_manual_PQK, rho_matrices


def K_FQK(fmap, X_distribution, n_qubits, X_distribution_test = None):
    if X_distribution_test is None: #K(X_distribution, X_distribution)
        rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)
    
        K_manual = np.zeros((len(X_distribution), len(X_distribution)))
        for i in range(len(X_distribution)):
            for j in range(len(X_distribution)):
                K_manual[i, j] = np.real(np.trace(rho_matrices_train[i] @ rho_matrices_train[j]))
        rho_matrices = rho_matrices_train

    else: #K(X_distribution, X_distribution_test)
        rho_matrices_train = get_rho_matrices(fmap, X_distribution, n_qubits)
        rho_matrices_test = get_rho_matrices(fmap, X_distribution_test, n_qubits)
        
        K_manual = np.zeros((len(X_distribution), len(X_distribution_test)))
        for i in range(len(X_distribution)):
            for j in range(len(X_distribution_test)):
                K_manual[i, j] = np.real(np.trace(rho_matrices_train[i] @ rho_matrices_test[j]))
        rho_matrices = [rho_matrices_train, rho_matrices_test]
    return K_manual, rho_matrices

def K_PQK_with_different_gamma(K_original, gamma_original, gamma_new):
    F = np.log(K_original)/(-gamma_original)
    K_new = np.exp(-gamma_new*F)
    return K_new





def variance_off_diagonal(M):
    """
    Remove main diagonal of M, transform M to a 1D array and calculates the variance of this array.
    """
    #removes main diagonal:
    M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1)
    M = M.flatten()
    variance = np.var(M)
    return variance



#################################33 Separable Rx ############################################


def manual_separable_rx(x_vector, y_vector, c):
    """This function computes the kernel value for the separable_rx circuit.
    """
    num_qubits = len(x_vector)
    K_value = 1
    for i in range(num_qubits):
        K_value *= np.cos(c*(x_vector[i] - y_vector[i])/2)**2
    return K_value




    
factorial_list_upto_10 = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])



def rbf_poly_approximation(X, Y, gamma, degree):
    from sklearn.metrics.pairwise import euclidean_distances
    order = int(degree)
    def taylor_expand(M, order):
        return np.sum([M**i / factorial_list_upto_10[i] for i in range(order+1)], axis=0)

    r = -gamma*euclidean_distances(X, Y, squared=True)
    return taylor_expand(r, order)
    

def separable_rx_gram_matrix(X, Y, c, method = "FQK"):
    """This function computes the gram matrix for the separable_rx circuit."""
    X = np.array(X)
    Y = np.array(Y)
    n, m = X.shape[0], Y.shape[0]
    n_dimensions = X.shape[1]


    # Reshape X and Y to facilitate broadcasting
    X_reshaped = X[:, np.newaxis, :]
    Y_reshaped = Y[np.newaxis, :, :]

    # Compute the gram matrix using vectorized operations
#    K = np.prod(np.cos(c * (X_reshaped - Y_reshaped) / 2) ** 2, axis=-1)
    if method == "FQK": 
        K = np.prod(np.cos(c * (X_reshaped - Y_reshaped) / 2) ** 2, axis=-1)
    else:
        gamma=1
        #print(n_dimensions)
        #print(np.sum(np.cos(c * (X_reshaped - Y_reshaped)), axis=-1))
        K = np.exp(-2*gamma*n_dimensions)*np.exp(2*gamma*np.sum(np.cos(c * (X_reshaped - Y_reshaped)), axis=-1))

    return K

def cosine_expansion_term(x, n):  
    """
    x: (num_of_samples, num_of_dimensions)
    n: (num_of_terms, 1)
    """  
    return (-1) ** n / factorial(2 * n) * np.power(x, 2 * n)

def K_polynomial(delta_list, c, n):
    """ 
    Use the other implementation for the polynomial approximation
    """
    order_list = np.arange(n) # array of size n
    number_of_terms, number_of_samples = delta_list.shape
    terms = np.zeros((number_of_terms, number_of_samples))
    for i in range(number_of_terms):
        for j in range(number_of_samples):
            terms[i,j] = np.sum(cosine_expansion_term(0.5*c*(delta_list[i,j]), order_list))**2
    return np.prod(terms, axis=1)

def K_separable_rx_polynomial_fast(delta_list, c, order_list):
    """
    K(x,y) = cos(c/2*delta_1)^2 = 1/2 + 1/2*cos(c*delta_1) = 1/2 + 1/2*sum_{n=0}^{N} (-1)^(n) * c**(2n) * delta_1**(2n) / (2n)!.

    Order refers to the number of terms in the polynomial expansion of cosine.

    delta_list: (num_of_samples, num_of_dimensions)
    c: scalar
    order_list: if int then order_list = np.arange(order_list), else array which contains the orders of the polynomial kernel. For order = 3, then order_list = [0,1,2]
    return: (num_of_samples,)
    """
    if type(order_list) == int or type(order_list) == float:
        order_list = np.arange(int(order_list)+1)
    print("order_list", order_list)
    delta_list = delta_list[:, np.newaxis, :] # (num_of_samples, 1, num_of_dimensions)
    n_list = order_list[:, np.newaxis] # (num_of_terms, 1)
    expansion_term_tensor = cosine_expansion_term(c * (delta_list), n_list) # (num_of_samples, num_of_terms, num_of_dimensions)
    expansion_term_summed_tensor = np.sum(expansion_term_tensor, axis = 1)*0.5 + 0.5 # (num_of_samples, num_of_dimensions)

    return np.prod(expansion_term_summed_tensor, axis = 1) # (num_of_samples,)

def K_polynomial_fast(delta_list, c, order_list):
    """
    K(x,y) = cos(c/2*delta_1)^2 = 1/2 + 1/2*cos(c*delta_1) = 1/2 + 1/2*sum_{n=0}^{N} (-1)^(n) * c**(2n) * delta_1**(2n) / (2n)!.

    Order refers to the number of terms in the polynomial expansion of cosine.

    delta_list: (num_of_samples, num_of_dimensions)
    c: scalar
    order_list: if int then order_list = np.arange(order_list), else array which contains the orders of the polynomial kernel. For order = 3, then order_list = [0,1,2]
    return: (num_of_samples,)
    """
    if type(order_list) == int:
        order_list = np.arange(order_list+1)
    tensor = (delta_list*c)**order_list[-1] # (num_of_samples, num_of_dimensions)
    return np.prod(tensor, axis = 1) # (num_of_samples,)

def pairwise_difference(X, Y):
    """
    X: (num_of_samples1, num_of_dimensions)
    Y: (num_of_samples2, num_of_dimensions)

    return: (num_of_samples1*num_of_samples2, num_of_dimensions)

    Calculate the pairwise differences between each element of X and Y

    """
    differences = X[:, np.newaxis, :] - Y[np.newaxis, :, :] # (num_of_samples1, num_of_samples2, num_of_dimensions)
    #stack the differences
    differences = differences.reshape(-1, differences.shape[2]) # (num_of_samples1*num_of_samples2, num_of_dimensions)
    
    return differences

#from scipy import factorial

#from sklearn import pairwise distance

from sklearn.metrics.pairwise import pairwise_kernels

from scipy.special import factorial    


def pairwise_minus(X,Y):
    """
    X: (num_of_samples1, num_of_dimensions)
    Y: (num_of_samples2, num_of_dimensions)

    return: (num_of_samples1, num_of_samples2)

    Calculate the pairwise differences between each element of X and Y

    """
    differences = X[:, np.newaxis, :] - Y[np.newaxis, :, :] # (num_of_samples1, num_of_samples2, num_of_dimensions)
    
    #calculate the euclidean norm of the differences
    differences = np.linalg.norm(differences) # (num_of_samples1, num_of_samples2)
    
    return differences


    
def separable_rx_gram_matrix_fast(X, Y, c, polynomial_approximation=None, noise = False, simple_poly = False):
    """This function computes the gram matrix for the separable_rx circuit."""
    X = np.array(X) # (num_of_samples1, num_of_dimensions)
    Y = np.array(Y)
    n, m = X.shape[0], Y.shape[0] 

    # Reshape X and Y to facilitate broadcasting
    X_reshaped = X[:, np.newaxis, :] # (num_of_samples1, 1, num_of_dimensions)
    Y_reshaped = Y[np.newaxis, :, :] # (1, num_of_samples2, num_of_dimensions)

    # Compute the gram matrix using vectorized operations
    if polynomial_approximation is not None:
        #time it 
        delta_list = pairwise_difference(X, Y)
        
        if simple_poly is False:
            K = K_separable_rx_polynomial_fast(delta_list, c, polynomial_approximation) # (num_of_samples1*num_of_samples2,
        elif simple_poly is True:
            K = polynomial_kernel(X*c, Y*c, degree = polynomial_approximation, gamma=  1) # (num_of_samples1, num_of_samples2)

        # Reshape K to the desired shape
        K = K.reshape(n, m)
    else:
        K = np.prod(np.cos(c * (X_reshaped - Y_reshaped) / 2) ** 2, axis=-1)
    return K


def manual_var(c, num_qubits, num_layers):
    c = c*num_layers
    EhXY = 0.5 - 1.0*np.cos(c)/c**2 + 1.0/c**2
    EhXYsquare = 0.375 - 1.0*np.cos(c)/c**2 - 0.0625*np.cos(2*c)/c**2 + 1.0625/c**2
    return EhXYsquare**num_qubits - (EhXY**num_qubits)**2

def manual_var_off_diag_n(c, n):
    EhXYsquare = (0.375*c**2 + 0.125*np.sin(c)**2 + 1.0*np.cos(c) - 1.0)/c**2
    EhXY = (-0.5*c**2 - 1.0*np.cos(c) + 1.0)/c**2
    return EhXYsquare**1 - (EhXY**(2))

#Analytical solution

def varK_cosine(c, num_qubits, num_layers):
    if num_layers > 1:
        c = c*num_layers
    #if c == 1:
    #    return 7/64
    EhXYsquare = (16*(1 - 4*c**2)**2*np.sin(np.pi*c)**2 + (c**2 - 1)**2*(12*np.pi**2*c**2*(1 - 4*c**2)**2 + np.sin(2*np.pi*c)**2))/(32*np.pi**2*(4*c**5 - 5*c**3 + c)**2)
    EhXY = 0.5 + 0.5*np.sin(np.pi*c)**2/(np.pi**2*c**2*(c**2 - 1)**2)
    return EhXYsquare**num_qubits - (EhXY**num_qubits)**2       



def vark_large_c_limit(c, L, num_qubits, idx ):
    if idx == 0: #Cosine 
        a = 12
        b= 2**3
        p=32
    elif idx == 1:
        a = 12
        b= 2**3
        p=2**5
    s = a**num_qubits - b ** num_qubits
    return s/(p**num_qubits)