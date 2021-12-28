import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  

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
		print(f'The hamiltonian is H = J sum Z_i Z_i+1 +h sum X_i, with for i=1,...,{self.length} h={self.h} and J={self.J}')
		print('-----------------------------------------')

	def get_solution(self):
		'''
		Computes the analytical solution for the ground state energy
		Return : ground state energy E_0
		'''
		N = self.length
		if N > 20:
			print('System too big for computation')
			return

		X = np.array([[0,1], [1,0]], dtype=complex)
		Y = np.array([[0,-1j], [1j,0]], dtype=complex)
		Z = np.array([[1,0], [0,-1]], dtype=complex)
		I = np.array([[1,0], [0,1]], dtype=complex)

		XX = sp.kron(X, X)
		YY = sp.kron(Y, Y)
		ZZ = sp.kron(Z, Z)


		H = self.J*sp.kron(sp.kron(Z, sp.eye(2**(N-2)) ), Z) + self.h*sp.kron( sp.eye(2**(N-1) ), X)
		for i in range(0,N-1):
			A = self.J*sp.kron(sp.kron(sp.eye(2**i), ZZ), sp.eye(2**(N-2-i)) )
			B = self.h*sp.kron(sp.kron(sp.eye(2**i), X), sp.eye(2**(N-1-i)) )

			H += A + B

		E = sp.linalg.eigsh(H,k=1,return_eigenvectors=False,which='SA')

		return E[0]

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
	
	def __init__(self, length:int, J=1):
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
		print(f'The hamiltonian is H = J sum X_i X_i+1 + Y_i Y_i+1 + Z_i Z_i+1 for i=1,...,{self.length} with J={self.J}')
		print('-----------------------------------------')

	def get_solution(self):
		'''
		Computes the analytical solution for the ground state energy
		Return : ground state energy E_0
		'''
		N = self.length
		if N > 20:
			print('System too big for computation')
			return

		X = np.array([[0,1], [1,0]], dtype=complex)
		Y = np.array([[0,-1j], [1j,0]], dtype=complex)
		Z = np.array([[1,0], [0,-1]], dtype=complex)
		I = np.array([[1,0], [0,1]], dtype=complex)

		XX = sp.kron(X, X)
		YY = sp.kron(Y, Y)
		ZZ = sp.kron(Z, Z)


		H = sp.kron(sp.kron(X, sp.eye(2**(N-2)) ), X)+sp.kron(sp.kron(Y, sp.eye(2**(N-2)) ), Y)+sp.kron(sp.kron(Z, sp.eye(2**(N-2)) ), Z)
		for i in range(0,N-1):
			A = sp.kron(sp.kron(sp.eye(2**i), XX + YY + ZZ), sp.eye(2**(N-2-i)) )
			H += A

		H *= self.J
		E = sp.linalg.eigsh(H,k=1,return_eigenvectors=False,which='SA')

		return E[0]

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

		E = np.zeros( (n_conns,1) )
		E[0] = sample.T@np.roll(sample, -1)
		for i in range(1,n_conns):
			E[i] = 1 - sample[i-1]*sample[i%self.length]

		return self.J*E, s_prime, n_conns
