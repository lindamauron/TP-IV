'''
Simulate a chain of N spins s=+/-1 using MCMC with Metropolis-Hasting Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit

###############################################
# Functions 

@jit(nopython=True)
def energy(sample, J=1, h=0):
	'''
	Computes the energy based on Ising model E = - sum_nearest_neighboors S_i*S_j + h sum S_i
	Input:
	sample (1D array): spins of the sample 
	h (1D array): magnetic field
	J (2D array): interaction matrix

	Return : total energy (scalar)
	'''
	energy = 0
	for i in range(sample.size-1) : 
		energy -= J[i, i+1]*sample[i]*sample[i+1]
	return energy-J[0,-1]*sample[0]*sample[-1] + sum( np.multiply(h,sample) )


@jit(nopython=True)
def boltzmannn_unnormalized(sample, beta):
	'''
	Computes the unnormalized Boltzmann probability of the sample
	
	sample (1D array): spins of the system
	beta (scalar): inverse temperature of the system
	
	Return : unnormalized probability (scalar)
	'''
	return np.exp(-beta*energy(sample) )

@jit(nopython=True)
def partition_function(sample_size, beta):
	'''
	Computes the partition function of the Boltzmann probability 
	for a sample of a given size at given temperature

	sample_size (scalar): number of spins in the chain
	beta (scalar): inverse temperature of the system

	Return : partition function (scalar)
	'''
	return sum( 2*scipy.special.comb(sample_size, k)*np.exp(- beta*(2*k-sample_size) ) )


def boltzmannn(sample, beta):
	'''
	Computes the Boltzmann probability of the state of the sample

	sample_size (scalar): number of spins in the chain
	beta (scalar): inverse temperature of the system

	Return : boltzmann probability (scalar)
	'''

	return boltzmannn_unnormalized(sample, beta)/partition_function(sample.size, beta)

@jit(nopython=True)
def log_boltzmann_unnormalized(sample, beta):
	'''
	Computes the natural logarithm of the Boltzmann probability

	sample (1D array): spins of the system
	beta (scalar): inverse temperature of the system

	Return : log of Boltzmann probability
	'''
	return -beta*energy(sample)


###############################################
# Parameters
#Number of particles
n_spins = 500

#Inverse emperature
beta = 1e2 #[1/eV]

#Warm up
warm_up_iteration = 5000

#Number of loops to execute
n_loops = 50000


###############################################

#Creating vector of spins, randomly +1 or -1
sample = np.random.choice([-1.0, 1.0], n_spins)
#print(f'{sample}, {energy(sample)}')

energy_of_sample = np.zeros(n_loops)

for i in range(n_loops):
	#Choose randomly spin to flip
	spin_to_flip = np.random.randint(0,n_spins)
	new_sample = np.copy(sample)
	new_sample[spin_to_flip] = -new_sample[spin_to_flip]

	#Compute test (with unnormalized probability bc. of division btw both)
	R = np.exp( log_boltzmann_unnormalized(new_sample, beta) - log_boltzmann_unnormalized(sample, beta) )


	eta = np.random.uniform()

	if R > eta:
		sample = np.copy(new_sample)

	
	#Computing the energy of the sample
	energy_of_sample[i] = energy(sample)


###############################################
# Graphics
E_mean = np.sum(energy_of_sample)/n_loops

plt.figure()
plt.plot(energy_of_sample, 'o')
plt.axvline(x=warm_up_iteration, color='r')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.title(f"{n_spins} spins, {n_loops} loops, Mean energy = {E_mean}, beta = {beta}")


#Compute density of states
k = np.arange(0,n_spins,2)  #number of unparallel pairs
E_theoretical = 2*k-n_spins #energies
#DOS = 2*scipy.special.comb(n_spins, k)*np.exp(-beta*E_theoretical)/partition_function(n_spins, beta)


plt.figure()
plt.hist(energy_of_sample[warm_up_iteration:], bins='auto')
#plt.plot(E_theoretical, (n_loops-warm_up_iteration)*DOS, 'r-o')
plt.xlabel('Energy')
plt.ylabel('Occurences')
plt.title(f"{n_spins} spins, {n_loops} loops, beta = {beta}")
plt.show();


