import numpy as np

class Model:
	'''
	Abstract structure (non-instanciable) of a model
	'''

	def log_psi(self, sample):
		'''
		Logarithm of the wavefunction
		Input : 
		sample (1D array) : vector basis of spins
		Return : log( psi(s)) (complex)
		'''
		raise NotImplementedError

	def log_prob(self, sample):
		raise NotImplementedError

	def local_energy(self, hamiltonian, sample):
		H, s_prime, n_conns = hamiltonian.get_H_terms(sample)

		E = 0
		for i in range(n_conns):
			E += H[i]*np.exp( self.log_psi(s_prime[i]) - self.log_psi(sample) )

		return E





class MeanField(Model):

	def __init__(self, length):
		'''
		Initialization
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		'''
		self.name = "Mean Field"
		self.length = length
		self.parameters = (1+1j)*np.ones( (length,1), dtype=complex)

	def log_psi(self, sample):
		return self.parameters.T@sample

	def log_prob(self, sample):
		return ( self.parameters+np.conj(self.parameters) ).T@sample


	def local_gradient(self, hamiltonian, sample):
		H, s_prime, n_conns = hamiltonian.get_H_terms(sample)

		grad = np.zeros( (self.length,1), dtype=complex )
		for i in range(n_conns):
			s = np.reshape(s_prime[i], (self.length,1) )
			grad += H[i]*np.exp( self.log_psi(s)-self.log_psi(sample) )*(s - sample)

		return grad

	def gradient(self, hamiltonian, list_of_samples):
		grad = np.zeros( (self.length,1), dtype=complex )

		for i in range(list_of_samples.shape[0]):
			grad += self.local_gradient(hamiltonian, list_of_samples[i])

		grad /= list_of_samples.shape[0]

		return grad

class Jastrow(Model):

	def __init__(self, length):
		'''
		Initialization
		n_samples (int): number of samples
		parameters (1D array): parameters {b_i} of the system
		'''
		self.name = "Jastrow"
		self.length = length
		self.parameters = (1+1j)*np.triu( np.ones( (length,length), dtype=complex), k=1)


	def log_psi(self, sample):
		return sample.T@self.parameters@sample

	def log_prob(self, sample):
		return sample.T@( self.parameters+np.conj(self.parameters) )@sample


	def local_gradient(self, hamiltonian, sample):
		H, s_prime, n_conns = hamiltonian.get_H_terms(sample)

		grad = np.zeros( (self.length,self.length), dtype=complex )

		for k in range(n_conns):
			s = np.reshape(s_prime[k], (self.length,1) )
			for i in range(self.length):
				for j in range(i+1, self.length):
					grad[i,j] += H[k]*np.exp( self.log_psi(s)-self.log_psi(sample) )*(sample[i]*sample[j]-s[i]*s[j])

		return grad

	def gradient(self, hamiltonian, list_of_samples):
		grad = np.zeros( (self.length,self.length), dtype=complex )

		for i in range(list_of_samples.shape[0]):
			grad += self.local_gradient(hamiltonian, list_of_samples[i])

		grad /= list_of_samples.shape[0]

		return np.reshape(grad,(self.length,self.length) )