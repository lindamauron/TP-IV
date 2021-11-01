'''
Simulate a chain of N spins s=+/-1 using Variational mean field model and MCMC with Metropolis-Hasting Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit

###############################################
# Functions 

#@jit(nopython=True)
def create_J(size, type_of_J='nearest_neighboors'):
	'''
	Creates the interaction matrix J for the 1D chain
	size (int): number of interacting spins (give matrix of size X size)
	type_of_j (string): type of interaction defined by J
						'zero' : J=0
						'nearest_neighboors' : J!=0 if two neighboors (with same intensity)

	Return : J(2D matrix) : interaction matrix
	'''
	J = np.zeros( (size, size) )

	if type_of_J=='zero':
		return J
	elif type_of_J == 'nearest_neighboors':
		for i in range(1,size-1):
			J[i, i-1] = 1
			J[i, i+1] = 1
		J[0, -1] = 1
		J[0, 1] = 1
		J[-1, 0] = 1
		J[-1, -2] = 1

	return J


def create_h(size, type_of_h='homogeneous', position=None):
	'''
	Creates the magnetic field vector h
	size (int): number of spins (gives vector of sizeX1)
	type_of_h (string): type of magnetic field defined by h
						'zero' : h=0
						'homogeneous' : h=1
						'peak' : h[position] = 10, otherwise h=1

	Return : J(2D matrix) : interaction matrix
	'''
	if type_of_h=='homogeneous':
		return np.ones(size)
	elif type_of_h=='zero':
		return np.zeros(size)
	elif type_of_h=='peak' and position != None:
		h = np.ones(size)
		for p in position:
			h[p] = 10

	return h


def trivial_energy(sample, parameters):
	'''
	Computes the energy based on Ising model E = - sum b_k s_k
	Input:
	sample (1D array): spins {sigma_i} of the sample 
	parameters (1D array): parameters {b_i} of the system

	Return : total energy (scalar)
	'''
	return -sum(sample*parameters)

def trivial_unnormalized_log_probability(sample, parameters, beta):
	'''
	Computes the log probability of the unnormalized trivial energy
	Input:
	sample (1D array): spins {sigma_i} of the sample 
	parameters (1D array): parameters {b_i} of the system
	beta (scalar): inverse temperature of the system
	
	Return : unnormalized logarithm of probability(scalar)
	'''
	return -beta*trivial_energy(sample, parameters)

def trivial_free_energy(parameters, beta):
	'''
	Computes the free energy F_tr = sum_spins exp(b E(spins)) 
	Input : 
	parameters (1D array): parameters {b_i} of the system
	beta (scalar): inverse temperature of the system

	Return : free energy of trivial approximation (scalar)
	'''
	F = 0
	for b in parameters:
		F += np.log( np.cosh(beta*b) )
	
	return -( F + parameters.size*np.log(2) )/beta


def mean_spin(sample, parameters, beta, index):
	'''
	Computes the mean of sigma[index] in the trivial energy
	Input : 
	sample (1D array): spins {sigma_i} of the sample 
	parameters (1D array): parameters {b_i} of the system
	beta (scalar): inverse temperature of the system
	index (integer): i

	Return : mean of sigma_index (scalar)
	'''
	return np.tanh( beta*parameters[index] )


def free_energy(sample, parameters, beta):
	'''
	Computes the free energy of the system
	Input : 
	sample (1D array): spins {sigma_i} of the sample 
	parameters (1D array): parameters {b_i} of the system
	beta (scalar): inverse temperature of the system


	Return : free energy (scalar)
	'''
	F = trivial_free_energy(parameters, beta)

	# Pre-compute the mean spins
	sigma = [mean_spin(sample, parameters, beta, i) for i in range(sample.size)]

	for i in range(sample.size-1):
		F += (- J[i,i+1]*sigma[i+1] + (h[i]-parameters[i]) ) * sigma[i]

	return F+(- J[0,-1]*sigma[0] + (h[-1]-parameters[-1]) ) * sigma[-1]


def new_parameter(beta, parameters, h, J, index):
	'''
	Computes the variational new parameter b[index]
	Input : 
	beta (scalar): inverse temperature of the system
	parameters (1D array): parameters {b_i} of the system
	h (1D array): magnetic field
	J (2D array): interaction matrix
	index (int): i

	Return : parameter[index] for new step (scalar)
	'''
	if index == 0:
		return h[0] + J[0,-1]*np.tanh(beta*parameters[-1]) + J[0,1]*np.tanh(beta*parameters[1])
	elif index == parameters.size-1:
		return h[-1] + J[-1,index-1]*np.tanh(beta*parameters[index-1]) + J[-1,0]*np.tanh(beta*parameters[0])
	else :
		return h[index] + J[index,index-1]*np.tanh(beta*parameters[index-1]) + J[index,index+1]*np.tanh(beta*parameters[index+1])

def MCMC(sample, parameters, h, J, beta, n_loops=1000):
	'''
	Executes the MCMC algorithm
	Input :
	sample (1D array): spins {sigma_i} of the sample 
	parameters (1D array): parameters {b_i} of the system
	h (1D array): magnetic field
	J (2D array): interaction matrix
	beta (scalar): inverse temperature of the system
	n_loops(int) : number of MCMC loops to exxecute

	Return : sample(1D array)
	'''
	n_spins = sample.size


	energy_of_sample = np.zeros(n_loops)

	for i in range(n_loops):
		#Choose randomly spin to flip
		spin_to_flip = np.random.randint(0,n_spins)
		new_sample = np.copy(sample)
		new_sample[spin_to_flip] = -new_sample[spin_to_flip]

		#Compute test (with unnormalized probability bc. of division btw both)
		R = np.exp( trivial_unnormalized_log_probability(new_sample, parameters, beta) 
			- trivial_unnormalized_log_probability(sample, parameters, beta) )

		eta = np.random.uniform()

		if R > eta:
			sample = np.copy(new_sample)


		energy_of_sample[i] = trivial_energy(sample, parameters)
	
	plt.figure()
	plt.hist(energy_of_sample, bins='auto')
	plt.xlabel('Energy')
	plt.ylabel("Occurences")
	#plt.title(f"{n_spins} spins, {n_loops} loops, Mean energy = {E_mean}, beta = {beta}")
	

	return sample


######### ######################################
# Parameters
#Number of particles
n_spins = 20

#Temperature
beta = 1e-2 #[eV]

#Warm up
warm_up_iteration = 5000

#Number of loops to execute
#n_loops = 50000

n_variational_loops = 10

J = create_J(n_spins, 'nearest_neighboors')
h = create_h(n_spins, 'zero')

#print(J)
#print('On fait des tests')
#print(h)
###############################################
# Variational computation w.r.t. parametesr {b_i}

parameters = np.zeros( (n_variational_loops+1, n_spins) )


for i in range(n_variational_loops):
	if i == 0:
		#Creating vector of spins, randomly +1 or -1
		sample = np.random.choice([-1.0, 1.0], n_spins)
		#print(f'{sample}, {energy(sample)}')

		parameters[i,:] = np.ones(n_spins)

		#Warm-up iterations
		sample = MCMC(sample, parameters[i,:], h, J, beta, warm_up_iteration)

	# do MCMC with given parameters for the probability
	sample = MCMC(sample, parameters[i,:], h, J, beta, n_loops=5000)

	# Change paramters according to grad F = 0
	parameters[i+1,:] = [new_parameter(beta, parameters[i,:], h, J, k) for k in range(n_spins) ]


print(f'b is {parameters[-1,:]}')
###############################################

plt.figure()
plt.plot(range(n_variational_loops+1), parameters)
plt.xlabel('Iteration')
plt.ylabel('Parameters b_i')
plt.legend( [f'$b({i})$' for i in range(n_spins)] )
plt.show();

