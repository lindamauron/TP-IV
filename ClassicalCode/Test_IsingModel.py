import numpy as np
import matplotlib.pyplot as plt
import itertools 
import scipy.special

import Models

beta = 1e0

'''
N = 10
simple = Models.ExactIsing1D(beta, N, h=0)
Z_simple = simple.partition_function
print(Z_simple)
'''

'''


n_spins = 20
model_test_partition = Models.ExactIsing1D(beta, n_spins, h=0)
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
	model_varying_beta = Models.ExactIsing1D(betas[k], 50) 
	E, DOS = model_varying_beta.DOS()

	E_varying_beta[k,:] = np.array(E)
	density_of_states[k,:] = np.array(DOS)

plt.figure()
plt.plot(E_varying_beta.T, density_of_states.T)
plt.xlabel('Energy')
plt.ylabel('DOS')
plt.legend( [betas[k] for k in range(betas.size)] , loc='upper right')
plt.title('Varying beta')
'''

Z_transf = []
Z_spins = []
Z_E = []
n_array = np.array([3,5,8,10,15,20,25])
for n in n_array:
	print(n)
	test_partition = Models.ExactIsing1D(beta=beta, n_samples=n, h=0)
	Z_transf.append(test_partition.partition_function)


	s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=n)))
	#print(s_tuples)
	Z = 0
	for s in s_tuples:
		Z += np.exp(-beta*test_partition.energy(s))
	Z_spins.append(Z)


	k = np.arange(0,n,2)
	Z_E.append( sum( 2*scipy.special.comb(n, k)*np.exp(- beta*(2*k-n) ) ) )

Z_transf = np.array(Z_transf)
Z_spins = np.array(Z_spins)
Z_E = np.array(Z_E)


plt.figure()
#plt.plot(n_array, Z_spins, label='Sum over spins')
plt.plot(n_array, Z_E/Z_spins, label='Sum over energies')
plt.plot(n_array, Z_transf/Z_spins, label='Transfer matrix')
plt.xlabel(r'$N$')
plt.ylabel(r'$Z$')
plt.legend()




plt.show()

