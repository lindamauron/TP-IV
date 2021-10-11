'''
Simulate a chain of N spins s=+/-1 using MCMC with Metropolis-Hasting Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

###############################################
# Functions

def energy(sample):
	'''
	Computes the energy based on Ising model E = - sum_nearest_neighboors S_i*S_j
	
	sample (1D array): spins of the sample 
	Return : total energy (scalar)
	'''
	energy = 0
	for i in range(sample.size-1) : 
		energy -= sample[i]*sample[i+1]
	return energy-sample[0]*sample[-1]

def boltzmannn_unnormalized(sample, temperature):
	'''
	Computes the unnormalized Boltzmann probability of the sample
	
	sample (1D array): spins of the system
	temperature (scalar): temperature of the system (in [eV], i.e. times the boltzmannn constant)
	
	Return : unnormalized probability (scalar)
	'''
	return np.exp(-energy(sample)/temperature)

def partition_function(sample_size, temperature):
	'''
	Computes the partition function of the Boltzmann probability 
	for a sample of a given size at given temperature

	sample_size (scalar): number of spins in the chain
	temperature (scalar): temperature of the system (in [eV], i.e. times the boltzmannn constant)

	Return : partition function (scalar)
	'''
	return np.power(2*np.cosh(1/temperature), sample_size) + np.power(2*np.sinh(1/temperature), sample_size)

def boltzmannn(sample, temperature):
	'''
	Computes the Boltzmann probability of the state of the sample

	sample_size (scalar): number of spins in the chain
	temperature (scalar): temperature of the system (in [eV], i.e. times the boltzmannn constant)

	Return : boltzmann probability (scalar)
	'''

	return boltzmannn_unnormalized(sample, temperature)/partition_function(sample.size, temperature)

###############################################
# Parameters
#Number of particles
n_spins = 500

#Temperature
temperature = 1.0 #[eV]

#Warm up
warm_up_iteration = 1500

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

	#Compute test
	R = np.exp(-(energy(new_sample) - energy(sample))/temperature)

	eta = np.random.uniform()

	if R > eta:
		sample = np.copy(new_sample)

	
	#Computing the energy of the sample
	energy_of_sample[i] = energy(sample)


###############################################
# Graphics
E_moy = np.sum(energy_of_sample)/n_loops

plt.figure()
plt.plot(energy_of_sample, 'o')
plt.plot(warm_up_iteration, energy_of_sample[warm_up_iteration], 'rx')
plt.title(f"{n_spins} spins, {n_loops} loops, Mean energy = {E_moy}")
plt.xlabel('Iteration')
plt.ylabel("Energy")



k = (energy_of_sample+n_spins)/2.0
DOS = 2*scipy.special.comb(n_spins, k)*np.exp(-energy_of_sample/temperature)/partition_function(n_spins, temperature)
plt.figure()
plt.hist(energy_of_sample[warm_up_iteration:], bins='auto')
plt.plot(energy_of_sample, n_loops*DOS, 'r-o')
plt.title(f"{n_spins} spins, {n_loops} loops")
plt.ylabel('Occurences')
plt.show()
