import numpy as np


cache = {}
def subspace_dimension(t, N):
    """
        d^(t)_sym = (N + t - 1)! / (t! * (N - 1)!)

        Return the dimension of the subspace of the t-fold tensor product of N qubits.

        See Expressibility and Entanglement Capability paper. See paragraph after Eq 12
    """
    #old    return np.math.factorial(int(N + t - 1) ) / (np.math.factorial(int(t)) * np.math.factorial(int(N - 1)))
    t_list = np.arange(1, t+1)
    integer_list = np.arange(N, N+t)
    return np.prod(integer_list,dtype=np.int64) / np.prod(t_list, dtype=np.int64)


def haar_frame_potential(t,N):
    """

        F^t_{Haar} = 1/d^(t)_sym

        Return the frame potential of the Haar integral for t-fold tensor product of N qubits.
        See Expressibility and Entanglement Capability paper. See Eq 10 and paragraph after Eq 12 
    """
    if (t,N) not in cache:
        cache[(t,N)] = 1/subspace_dimension(t,N)
    return cache[(t,N)]

def A_expressibility(K_fidelity,  num_qubits, t = 2 ):
    """
        Return the trace norm of expressibility metric A for the t-fold tensor product of N qubits.
        See Expressibility and Entanglement Capability paper. This is still not clear if correctly done

        See overleaf proof
    """
    return np.sqrt((np.mean(K_fidelity**t) - haar_frame_potential(t, int(2**num_qubits))))


















