import numpy as np


class IsingTransverse:
	'''
	Transverse Ising model H = J sum Z_i Z_i+1 +h sum X_i
	'''
	
	def __init__(self, length, J=-1, h=-1):
		'''
		Initialization
		n_samples (int): number of samples
		J (scalar): type of interaction defined by J
					J*nearest neighboors interaction
		h (scalar): type of magnetic field defined by h
					h*homogeneous
		'''
		self.length = length
		self.h = h
		self.J = J

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'The hamiltonian is H = J sum Z_i Z_i+1 +h sum X_i, with h={self.h} and J={self.J}')
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

	
class Heisenberg:
	'''
	Heisenberg model H = J sum X_i X_i+1 + Y_i Y_i+1 + Z_i Z_i+1
	'''
	
	def __init__(self, length:int, J=-1):
		'''
		Initialization
		n_samples (int): number of samples
		J (scalar): type of interaction defined by J
		'''
		self.length = length
		self.J = J

	def print_infos(self):
		'''
		Prints all info relative to the class
		'''
		print(f'The hamiltonian is H = J sum X_i X_i+1 + Y_i Y_i+1 + Z_i Z_i+1 with J={self.J}')
		print('-----------------------------------------')

	def get_conns(self, sample):
		'''
		Computes all s' in the local energy sum, the corresponding energy term and their number
		'''
		s_list = np.repeat( sample, self.length+1, axis=1 ).T
		for i in range(1,self.length+1):
			s_list[i,i-1] *= -1
			s_list[i,i%self.length] *= -1

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

		E = self.J*np.ones( (n_conns,1) )
		E[0] = sample.T@np.roll(sample, -1)
		for i in range(1,n_conns):
			E[i] *= 1 + sample[i-1]*sample[i%self.length]

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

