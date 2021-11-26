import numpy as np


class IsingTransverse:
	'''
	Transverse Ising model H = J sum Z_i Z_i+1 +h sum X_i
	'''
	
	def __init__(self, beta, length, J=-1, h=-1):
		'''
		Initialization
		beta (scalar): inverse temperature of the system
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		J (scalar): type of interaction defined by J
					J*nearest neighboors interaction
		h (scalar): type of magnetic field defined by h
					h*homogeneous
		'''
		self.beta = beta
		self.length = length
		self.h = h
		self.J = J

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'Inverse temperature of the system : {self.beta}')
		#print(f'Partition function : {self.partition_function}')
		print(f'The transverse field is h={self.h} and the interactions J={self.J}')
		print('-----------------------------------------')

	def get_conns(self, sample):
		'''
		Computes all s' in the local energy sum, the corresponding energy term and their number
		'''
		s_list = np.repeat( sample, self.length+1, axis=1 ).T
		for i in range(1,self.length+1):
			s_list[i,i-1] = -s_list[i,i-1]

		return s_list, self.length+1

	def get_H_terms(self, sample):
		'''
		Gives all the therms which appear for the local energy calculation
		sample (1D array) : s
		Return : E (vector) energy corresponding to each sample
				s_prime (matrix) : list of samples
				n_conns (scalar) : number of s_prime
		'''
		s_prime, n_conns = self.get_conns(sample)

		E = self.h*np.ones( (n_conns,1) )
		E[0] = self.J*sample.T@np.roll(sample, -1)

		return E, s_prime, n_conns

	


def X(i, sample):
	''''
	Pauli matrix X on spin i
	'''
	s = sample
	s[i] = -s[i]
	return s

def Y(i, sample):
	''''
	Pauli matrix Y on spin i
	'''
	s = sample
	s[i] = -s[i]
	return 1j*s[i]*s

def Z(i, sample):
	''''
	Pauli matrix Z on spin i
	'''
	return sample[i]*sample

