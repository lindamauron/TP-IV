import numpy as np
from numba import jit
import scipy.special


class ExactIsing1D:
	'''
	Computes the quantities in the exact Ising in 1D H = - sum J_k,k+1 s_k*s_k+1 + sum h_k s_k
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
		self.J = self.create_J(type_of_J)
		self.h = self.create_h(type_of_h)
		self.partition = None

	#@jit(nopython=True)
	def create_J(self, type_of_J):
		'''
		Creates the interaction matrix J for the 1D chain
		type_of_j (string): type of interaction defined by J
							'zero' : J=0
							'nearest_neighboors' : J!=0 if two neighboors (with same intensity)

		Return : J(2D matrix) : interaction matrix
		'''
		J = np.zeros( (self.n_samples, self.n_samples) )

		if type_of_J=='zero':
			return J
		elif type_of_J == 'nearest_neighboors':
			for i in range(1,self.n_samples-1):
				J[i, i-1] = 1
				J[i, i+1] = 1
			J[0, -1] = 1
			J[0, 1] = 1
			J[-1, 0] = 1
			J[-1, -2] = 1

		return J

	def create_h(self, type_of_h):
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
			return np.ones(self.n_samples)
		elif type_of_h=='zero':
			return np.zeros(self.n_samples)

		return h

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'Inverse temperature of the system : {self.beta}')
		#print(f'Partition function : {self.partition_function}')
		print(f'The field is {self.h} and the interactions \n {self.J}')
		print('-----------------------------------------')


	########################################
	# Statistical quantities

	def energy(self, sample):
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
			energy -= self.J[i, i+1]*sample[i]*sample[i+1]
		return energy-self.J[0,-1]*sample[0]*sample[-1] - sum( np.multiply(self.h,sample) )

	def boltzmannn_unnormalized(self, sample):
		'''
		Computes the unnormalized Boltzmann probability of the sample
		Input:
		sample (1D array): spins of the system
		beta (scalar): inverse temperature of the system
		
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

			k = np.arange(0,self.n_samples,2)
			self.partition = sum( 2*scipy.special.comb(self.n_samples, k)*np.exp(- self.beta*(2*k-self.n_samples) ) )
		
		return self.partition

	def boltzmannn(self, sample):
		'''
		Computes the Boltzmann probability of the state of the sample
		Input:
		sample_size (scalar): number of spins in the chain
		beta (scalar): inverse temperature of the system

		Return : boltzmann probability (scalar)
		'''

		return self.probability_unnormalized(sample)/self.partition_function

	def unnormalized_log_probability(self, sample):
		'''
		Computes the natural logarithm of the Boltzmann probability
		Input:
		sample (1D array): spins of the system
		beta (scalar): inverse temperature of the system

		Return : log of Boltzmann probability
		'''
		return -self.beta*self.energy(sample)

	def DOS(self):
		'''
		Computes the density of states
		
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
		J = self.J[0,1]
		h = self.h[0]

		b = np.exp(self.beta*J)*np.cosh(self.beta*h)
		sq = np.sqrt( np.exp(2*self.beta*J)*np.sinh(self.beta*h)**2 + np.exp(-2*self.beta*J) )

		l1 = b + sq
		l2 = b - sq

		l2_l1 = 2*np.sinh(2*self.beta*J)/l1**2

		return -self.n_samples*np.log(l1)/self.beta - np.log( 1+l2_l1**self.n_samples)/self.beta
