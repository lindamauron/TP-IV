import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit


def partition_function(sample_size, temperature):
	'''
	Computes the partition function of the Boltzmann probability 
	for a sample of a given size at given temperature

	sample_size (scalar): number of spins in the chain
	temperature (scalar): temperature of the system (in [eV], i.e. times the boltzmannn constant)

	Return : partition function (scalar)
	'''
	#Compute density of states
	k = np.arange(0,sample_size,2)  #number of unparallel pairs
	E_theoretical = 2*k-n_spins #energies
	
	return sum( 2*scipy.special.comb(n_spins, k)*np.exp(-E_theoretical/temperature) )


N = np.arange(3, 10, 1)

Z = [partition_function(n, 1.0) for n in N]
plt.figure()
plt.plot(N, Z)
plt.plot(N, np.power(2*np.cosh(1), N) + np.power(2*np.sinh(1), N))
plt.show()



