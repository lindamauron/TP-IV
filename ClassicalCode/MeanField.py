import numpy as np
from numba import jit
import ExactIsing1D


class MeanField:
	'''
	Computes the quantities in the Mean Field approximation H = - sum_k b_k s_k
	'''	

	########################################
	# Creation functions
	def __init__(self, beta, n_samples, type_of_J='nearest_neighboors', type_of_h='homogeneous'):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		type_of_j (string): type of interaction defined by J
						'zero' : J=0
						'nearest_neighboors' : J!=0 if two neighboors (with same intensity)
		type_of_h (string): type of magnetic field defined by h
						'zero' : h=0
						'homogeneous' : h=1
		'''
		self.beta = beta
		self.n_samples = n_samples
		self.parameters = np.ones(n_samples)

		self.exact_model = ExactIsing1D.ExactIsing1D(beta, n_samples, type_of_J, type_of_h)

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'Inverse temperature of the system : {self.beta}')
		print(f'The actual parameters are {self.parameters}')
		self.exact_model.print_infos()
		print('-----------------------------------------')


	########################################
	# Statistical quantities
	def energy(self, sample):
		'''
		Computes the energy of the system
		Input:
		sample (1D array): spins {sigma_i} of the sample 

		Return : total energy (scalar)
		'''
		return -sum( np.multiply(sample,self.parameters) )


	def unnormalized_log_probability(self, sample):
		'''
		Computes the log Boltzmann probability of the unnormalized energy
		Input:
		sample (1D array): spins {sigma_i} of the sample 
		
		Return : unnormalized logarithm of probability(scalar)
		'''
		return -self.beta*self.energy(sample)

	def partition_function(self):
		'''
		Computes the partition function of the Boltzmann probability 
		and stocks the value once used

		Return : partition function (scalar)
		'''
		Z = 2**self.n_samples
		for b in self.parameters:
			Z *= np.cosh(self.beta*b)

		return Z

	def log_probability(self, sample):
		'''
		Computes the log proabability of the sample
		Input :
		sample (1D array): spins {sigma_i} of the sample 

		Return : log probability (scalar)
		'''
		return self.unnormalized_log_probability(sample) - np.log(self.partition_function())

	def local_free_energy(self, sample):
		''' 
		Computes the local free energy F_loc
		Input :
		sample (1D array): spins {sigma_i} of the sample 

		Return : F_loc (scalar)
		'''
		return self.exact_model.energy(sample) + self.log_probability(sample)/self.beta
	
	########################################
	# Variational quantities

	def local_gradient(self, sample):
		'''
		Computes the gradient of F_loc
		Input :
		sample (1D array): spins {s_i} of the sample 

		Return : gradient (1D array)
		'''
		grad = np.zeros(self.n_samples)
		for k in range(self.n_samples):
			grad[k] = - np.tanh(self.beta*self.parameters[k]) + sample[k]

		return grad

	def gradient(self, list_of_samples):
		'''
		Computes the gradient of F_lambda
		Input:
		list_of_samples (2D array): all the samples on which to average

		Return : gradient (1D array)
		'''
		# Number of samples
		ns = list_of_samples[:,0].size

		# F_lambda
		f_lambda = 0
		for i in range(ns):
			f_lambda += self.local_free_energy(list_of_samples[i,:])
		f_lambda /= ns

		grad = 0
		for i in range(ns):
			s = list_of_samples[i,:]
			grad += self.beta*self.local_gradient(s)*(self.local_free_energy(s) - f_lambda)
		grad /= ns

		return grad
