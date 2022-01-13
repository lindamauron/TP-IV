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
		'''
		Logarithm of the state probability
		Input : 
		sample (1D array) : vector basis of spins
		Return : log( P(s)) (real)		
		'''
		raise NotImplementedError

	def local_energy(self, hamiltonian, sample):
		'''
		Computes the local energy E_loc(s)
		Input : 
		hamiltonian : hamiltonian to compute the energy
		sample (1D array) : vector basis of spins
		Return : E_loc(s) = sum_s' <s|H|s'> psi(s')/psi(s)
		''' 
		H, s_prime, n_conns = hamiltonian.get_H_terms(sample)

		E = 0
		for i in range(n_conns):
			E += H[i]*np.exp( self.log_psi(s_prime[i]) - self.log_psi(sample) )

		return E



class MeanField_sigmoid(Model):
	'''
	Psi(s) = prod_i sqrt( 1/(1+ exp(-theta s_i)) )
	'''

	def __init__(self, length):
		'''
		Initialization
		length (int): number of samples
		'''
		self.name = "Mean Field"
		self.length = length
		self.parameters = np.ones((1,1)) #np.random.normal()

	def log_psi(self, sample):
		'''
		Logarithm of the wavefunction
		Input : 
		sample (1D array) : vector basis of spins
		Return : log( psi(s)) (complex)
		'''
		return -0.5*np.log1p( np.exp(-self.parameters*sample) ).sum()

	def log_prob(self, sample):
		'''
		Logarithm of the state probability
		Input :
		sample (1D array) : vector basis of spins
		Return : log( P(s)) (real)		
		'''
		return -np.log1p( np.exp(-self.parameters*sample) ).sum()


	def gradient(self, hamiltonian, list_of_samples):
		'''
		Computes the gradient of the energy w.r.t. the parameters
		Input : 
		hamiltonian : hamiltonian to compute the energy
		list_of_samples (array (Ns,L,1)) : list of states to compute the expectation value
		Return : Gradient of the energy d<E> / dp_k for all p_k
		'''
		Ns = list_of_samples.shape[0]

		# Pre-computing the enrgies 
		E_loc = np.zeros( (Ns,1), dtype = complex )
		for i in range(Ns):
			E_loc[i] = self.local_energy(hamiltonian, list_of_samples[i])
		E = E_loc.mean()

		G = 0
		for i in range(Ns):
			# Derivative of the wave function
			D = 0.5*list_of_samples[i].T @ (1/(np.exp(self.parameters*list_of_samples[i])+1) )

			G += (E_loc[i] - E)*D/Ns

		return 2*np.real(G)


class Jastrow(Model):
	'''
	Psi(s) = Prod exp[J1 s_i s_i+1 + J2 s_i s_i+2]
	'''

	def __init__(self, length):
		'''
		Initialization
		length (int): number of samples
		'''
		self.name = "Jastrow"
		self.length = length
		self.parameters = -np.ones( (2,1) ) #np.random.normal( size=(2,1) )


	def log_psi(self, sample):
		'''
		Logarithm of the wavefunction
		Input : 
		sample (1D array) : vector basis of spins
		Return : log( psi(s)) (complex)
		'''
		return self.parameters[0]*sample.T@np.roll(sample, -1) + self.parameters[1]*sample.T@np.roll(sample, -2)

	def log_prob(self, sample):
		'''
		Logarithm of the state probability
		sample (1D array) : vector basis of spins
		Return : log( P(s)) (real)		
		'''
		return 2*self.log_psi(sample)


	def gradient(self, hamiltonian, list_of_samples):
		'''
		Computes the gradient of the energy w.r.t. the parameters
		Input : 
		hamiltonian : hamiltonian to compute the energy
		list_of_samples (array (Ns,L,1)) : list of states to compute the expectation value
		Return : Gradient of the energy d<E> / dp_k for all p_k
		'''
		Ns = list_of_samples.shape[0]

		# Pre-computing the enrgies 
		E_loc = np.zeros( (Ns,1), dtype = complex )
		for i in range(Ns):
			E_loc[i] = self.local_energy(hamiltonian, list_of_samples[i])
		E = E_loc.mean()


		G = np.zeros( (2,1,1), dtype=complex )
		for i in range(Ns):
			# Derivative of the wave function
			D1 = list_of_samples[i].T@np.roll(list_of_samples[i], -1)
			D2 = list_of_samples[i].T@np.roll(list_of_samples[i], -2)

			G += (E_loc[i] - E)*np.array([D1,D2])/Ns

		return 2*np.real(np.reshape(G, (2,1)))


class MeanField(Model):
	'''
	Psi(s) = prod_i exp(-theta s_i)
	'''

	def __init__(self, length):
		'''
		Initialization
		length (int): number of samples
		'''
		self.name = "Mean Field"
		self.length = length
		self.parameters = np.ones((1,1)) #np.random.normal()

	def log_psi(self, sample):
		'''
		Logarithm of the wavefunction
		Input : 
		sample (1D array) : vector basis of spins
		Return : log( psi(s)) (complex)
		'''
		return -self.parameters*sample.sum()

	def log_prob(self, sample):
		'''
		Logarithm of the state probability
		Input :
		sample (1D array) : vector basis of spins
		Return : log( P(s)) (real)		
		'''
		return 2*self.log_psi(sample)


	def gradient(self, hamiltonian, list_of_samples):
		'''
		Computes the gradient of the energy w.r.t. the parameters
		Input : 
		hamiltonian : hamiltonian to compute the energy
		list_of_samples (array (Ns,L,1)) : list of states to compute the expectation value
		Return : Gradient of the energy d<E> / dp_k for all p_k
		'''
		Ns = list_of_samples.shape[0]

		# Pre-computing the enrgies 
		E_loc = np.zeros( (Ns,1), dtype = complex )
		for i in range(Ns):
			E_loc[i] = self.local_energy(hamiltonian, list_of_samples[i])
		E = E_loc.mean()

		G = 0
		for i in range(Ns):
			# Derivative of the wave function
			D = -list_of_samples[i].sum()

			G += (E_loc[i] - E)*D/Ns

		return 2*np.real(G)


