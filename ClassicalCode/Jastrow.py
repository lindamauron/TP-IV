import numpy as np
from numba import jit
import ExactIsing1D
import itertools

class Jastrow:
	'''
	Computes the quantities in the Jastrow functions model H = - sum si Wij sj
	'''	

	########################################
	# Creation functions
	def __init__(self, beta, n_samples, type_of_J='nearest_neighboors', type_of_h='homogeneous'):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (2D array): parameters {W_ij} of the system (upper diagonal)
		type_of_j (string): type of interaction defined by J
						'zero' : J=0
						'nearest_neighboors' : J!=0 if two neighboors (with same intensity)
		type_of_h (string): type of magnetic field defined by h
						'zero' : h=0
						'homogeneous' : h=1
		'''
		self.beta = beta
		self.n_samples = n_samples
		self.parameters = np.triu( np.ones( (n_samples, n_samples) ), k=1)
		self.exact_model = ExactIsing1D.ExactIsing1D(beta, n_samples, type_of_J, type_of_h)

		#self.partition = 0
		#self.flag_new_parameters = True

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'Inverse temperature of the system : {self.beta}')
		print(f'The actual parameters are {self.parameters}')
		print(f'The exact model is :')
		self.exact_model.print_infos()
		print('-----------------------------------------')


	########################################
	# Statistical quantities
	def energy(self, sample):
		'''
		Computes the energy of the system
		Input:
		sample (1D array): spins {s_i} of the sample 

		Return : total energy (scalar)
		'''
		E = 0
		for i in range(self.n_samples):
			for j in range(self.n_samples):
				if i<j:
					E += sample[i]*self.parameters[i,j]*sample[j]
		return E

	def unnormalized_log_probability(self, sample):
		'''
		Computes the log Boltzmann probability of the unnormalized energy
		Input:
		sample (1D array): spins {s_i} of the sample 
		
		Return : unnormalized logarithm of probability(scalar)
		'''
		return -self.beta*self.energy(sample)

	def partition_function(self):
		'''
		Computes the partition function of the Boltzmann probability 
		and stocks the value once used

		Return : partition function (scalar)
		'''
		'''
		a = 1
		b = 1
		for i in range(self.n_samples):
			for j in range(self.n_samples):
				if i<j:
					a *= np.cosh(-self.beta*self.parameters[i,j])
					b *= np.sinh(-self.beta*self.parameters[i,j])

		Z = 2**self.n_samples*(a+b)
		return Z
		'''
		s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=self.n_samples)))
		Z = 0
		for s in s_tuples:
			Z += np.exp(-self.beta*self.energy(s))

		return Z
		


	def log_probability(self, sample):
		'''
		Computes the log proabability of the sample
		Input :
		sample (1D array): spins {s_i} of the sample 

		Return : log probability (scalar)
		'''
		return self.unnormalized_log_probability(sample) - np.log(self.partition_function())

	def local_free_energy(self, sample):
		''' 
		Computes the local free energy F_loc
		Input :
		sample (1D array): spins {s_i} of the sample 

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

		Return : gradient (2D array upper-diagonal)
		'''
		'''
		grad = np.zeros( (self.n_samples, self.n_samples) )
		a = 1
		b = 1

		# Compute denominator
		for i in range(self.n_samples):
			for j in range(self.n_samples):
				if i < j:
					a *= np.cosh(-self.beta*self.parameters[i,j])
					b *= np.sinh(-self.beta*self.parameters[i,j])

		# Gradient for W_kl
		for k in range(self.n_samples):
			for l in range(self.n_samples):
				if k < l:
					# WHAT IF 0 ?
					grad[k,l] = a*np.sinh(-self.beta*self.parameters[k,l])/np.cosh(-self.beta*self.parameters[k,l]) + b*np.cosh(-self.beta*self.parameters[k,l])/np.sinh(-self.beta*self.parameters[k,l])
					grad[k,l] /= (a+b)
					grad[k,l] -= sample[k]*sample[l]

		return grad
		'''
		df_dx = np.zeros( (self.n_samples, self.n_samples) )
		x0 = 1e-4
		f = self.local_free_energy(sample)
		for k in range(self.n_samples):
			for l in range(self.n_samples):
				if k<l:
					self.parameters[k,l] = self.parameters[k,l] + x0

					f_plus = self.local_free_energy(sample)

					df_dx[k,l] = (f_plus - f)/x0

					self.parameters[k,l] = self.parameters[k,l] - x0
		return -df_dx/f
		


	def gradient(self, list_of_samples):
		'''
		Computes the gradient of F_lambda
		Input:
		list_of_samples (2D array): all the samples on which to average

		Return : gradient (2D array) (upper-diagonal)
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
