import numpy as np
from numba import jit
import scipy.special
import itertools


class ExactIsing1D:
	'''
	Computes the quantities in the exact Ising in 1D H = - J sum s_k*s_k+1 - hsum s_k
	'''	

	########################################
	# Creation functions
	def __init__(self, beta, n_samples, J=1, h=1):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		J (scalar) : interaction between spins
		h (scalar) : external field
		'''
		self.beta = beta
		self.n_samples = n_samples
		self.J = J
		self.h = h
		self.partition = None


	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'# Inverse temperature of the system : {self.beta}')
		#print(f'Partition function : {self.partition_function}')
		print(f'# The Ising field is {self.h} and the interactions {self.J}')
		print('############################################')


	########################################
	# Statistical quantities

	def energy(self, sample):
		'''
		Computes the energy based on Ising model E = - J sum_i s_i*s_i+1 - h sum S_i
		Input:
		sample (1D array): spins of the sample 

		Return : total energy (scalar)
		'''
		energy = -self.J*sample.T @ np.roll(sample,-1)
		'''
		for i in range(self.n_samples) : 
			energy -= self.J*sample[i]*sample[ (i+1)%N ]
		'''
		return energy-self.h*sample.sum() 

	def probability_unnormalized(self, sample):
		'''
		Computes the unnormalized Boltzmann probability of the sample
		Input:
		sample (1D array): spins of the system
		
		Return : unnormalized probability (scalar)
		'''
		return np.exp(-self.beta*self.energy(sample) )

	@property
	def partition_function(self):
		'''
		Computes the partition function of the Boltzmann probability 
		and stocks the value once used

		Return : partition function (scalar)
		'''
		if self.partition is None:	
			J = self.J
			h = self.h

			b = np.exp(self.beta*J)*np.cosh(self.beta*h)
			sq = np.sqrt( np.exp(2*self.beta*J)*np.sinh(self.beta*h)**2 + np.exp(-2*self.beta*J) )

			l1 = b + sq
			l2 = b - sq

			self.partition = l1**self.n_samples + l2**self.n_samples
		return self.partition


	def probability(self, sample):
		'''
		Computes the Boltzmann probability of the state of the sample
		Input:
		sample (1D array): spins of the system

		Return : boltzmann probability (scalar)
		'''

		return self.probability_unnormalized(sample)/self.partition_function

	def unnormalized_log_probability(self, sample):
		'''
		Computes the natural logarithm of the Boltzmann probability
		Input:
		sample (1D array): spins of the system

		Return : log of Boltzmann probability
		'''
		return -self.beta*self.energy(sample)

	def DOS(self):
		'''
		Computes the density of states if h=0
		
		Return : energy spectrum (1D array) and corresponding occurences (1D array)
		'''
		k = np.arange(0,self.n_samples,2)  #number of unparallel pairs
		E = 2*k-self.n_samples #energies
		DOS = 2*scipy.special.comb(self.n_samples, k)*np.exp(-self.beta*E)/self.partition_function

		return E, DOS

	def free_energy(self):
		'''
		Computes the free energy of the system 
		for J=nearest_neighboors and h=homogeneous

		Return : Free energy F (scalar)
		'''
		J = self.J
		h = self.h

		b = np.exp(self.beta*J)*np.cosh(self.beta*h)
		sq = np.sqrt( np.exp(2*self.beta*J)*np.sinh(self.beta*h)**2 + np.exp(-2*self.beta*J) )

		l1 = b + sq
		l2 = b - sq

		l2_l1 = 2*np.sinh(2*self.beta*J)/l1**2

		return -self.n_samples*np.log(l1)/self.beta - np.log( 1+l2_l1**self.n_samples)/self.beta


class MeanField:
	'''
	Computes the quantities in the Mean Field approximation H = - sum_k b_k s_k
	'''	

	########################################
	# Creation functions
	def __init__(self, beta, n_samples, J=1, h=1):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		J (scalar) : interaction between spins
		h (scalar) : external field
		'''
		self.beta = beta
		self.n_samples = n_samples
		self.parameters = np.ones(n_samples)

		self.exact_model = ExactIsing1D(beta, n_samples, J, h)

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'# Inverse temperature of the system : {self.beta}')
		print(f'# The Mean Field parameters are {self.parameters}')
		self.exact_model.print_infos()
		print('############################################')


	########################################
	# Statistical quantities
	def energy(self, sample):
		'''
		Computes the energy of the system
		Input:
		sample (1D array): spins {sigma_i} of the sample 

		Return : total energy (scalar)
		'''
		return  -self.parameters.T@sample


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


class Jastrow:
	'''
	Computes the quantities in the Jastrow functions model H = W sum_i s_i s_i+1
	'''	

	########################################
	# Creation functions
	def __init__(self, beta, n_samples, J=1, h=1):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (2D array): parameters {W_ij} of the system (upper diagonal)
		J (scalar) : interaction between spins
		h (scalar) : external field
		'''
		self.beta = beta
		self.n_samples = n_samples
		self.parameters = 1
		self.exact_model = ExactIsing1D(beta, n_samples, J, h)


	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'# Inverse temperature of the system : {self.beta}')
		print(f'# The Jastrow parameters are {self.parameters}')
		print(f'# The exact model is :')
		self.exact_model.print_infos()
		print('############################################')


	########################################
	# Statistical quantities
	def energy(self, sample):
		'''
		Computes the energy of the system
		Input:
		sample (1D array): spins {s_i} of the sample 

		Return : total energy (scalar)
		'''
		return self.parameters*sample.T@np.roll(sample, -1)

	def partition_function(self):
		return 2**self.n_samples*( np.cosh(-self.beta*self.parameters)**self.n_samples + np.sinh(-self.beta*self.parameters)**self.n_samples)

	def unnormalized_log_probability(self, sample):
		'''
		Computes the log Boltzmann probability of the unnormalized energy
		Input:
		sample (1D array): spins {s_i} of the sample 
		
		Return : unnormalized logarithm of probability(scalar)
		'''
		return -self.beta*self.energy(sample)

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
		C = np.cosh(-self.beta*self.parameters)
		S = np.sinh(-self.beta*self.parameters)

		a = (C**(self.n_samples-2) + S**(self.n_samples-2) )/(C**self.n_samples + S**self.n_samples)
		a *= self.n_samples * C * S

		return a - sample.T@np.roll(sample, -1) 
		


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

