import numpy as np
import matplotlib.pyplot as plt
import itertools 
import scipy.special

import ExactIsing1D

beta = 1e0

'''
N = 10
simple = ExactIsing1D.ExactIsing1D(beta, N, type_of_h='zero')
Z_simple = simple.partition_function
print(Z_simple)
'''


# Verify exact solution of partitioon function when h=0 (for different sizes)
N = np.arange(3, 10, 1)
model_varying_n = np.array([ExactIsing1D.ExactIsing1D(beta, k, type_of_h='zero') for k in N])
Z_varying_n = np.array([model_varying_n[k].partition_function for k in range(N.size)])

plt.figure()
plt.plot(N, Z_varying_n)
plt.plot(N, np.power(2*np.cosh(beta), N) + np.power(2*np.sinh(beta), N))
plt.xlabel('N')
plt.ylabel('Partition function')
plt.title('Varying number of samples')



n_spins = 20
model_test_partition = ExactIsing1D.ExactIsing1D(beta, n_spins, type_of_h='zero')
Z_to_compare = model_test_partition.partition_function

s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=n_spins)))
print(s_tuples[0])

Z = np.exp(scipy.special.logsumexp( [model_test_partition.energy(s) for s in s_tuples]))
print(f'Z with itertools {Z} vs Z by combinatoric {Z_to_compare}')


# Check densities of states
betas = np.logspace(0, 0.5, 10)
print(betas)

E_varying_beta = np.zeros( (betas.size, 25) )
density_of_states = np.zeros( (betas.size, 25) )

for k in range(betas.size):
	model_varying_beta = ExactIsing1D.ExactIsing1D(betas[k], 50) 
	E, DOS = model_varying_beta.DOS()

	E_varying_beta[k,:] = np.array(E)
	density_of_states[k,:] = np.array(DOS)

plt.figure()
plt.plot(E_varying_beta.T, density_of_states.T)
plt.xlabel('Energy')
plt.ylabel('DOS')
plt.legend( [betas[k] for k in range(betas.size)] , loc='upper right')
plt.title('Varying beta')





plt.show()

