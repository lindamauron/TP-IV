import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit

class MCMC:
	'''Monte-Carlo with Markov Chains implementation'''

	def __init__(self, model, hamiltonian, iterations=5000):
		self.model = model
		self.hamiltonian = hamiltonian
		self.length = model.length
		self.iterations = iterations


	def print_infos(self):
		self.hamiltonian.print_infos()
		print(f'The model is : {self.model.name}')
		print(f'The number of Monte-Carlo iterations is {self.iterations}')
		print('-----------------------------------------')



	def run(self, sample=None):
		'''
		Executes the MCMC algorithm
		Input :
		sample (1D array): 	initial spins of the sample 
							if None, random sample
		Return : list of used samples (2D array)
		'''
		E_loc = np.zeros( (self.iterations, 1), dtype=complex )
		if sample is None:
			sample = np.random.choice([-1.0, 1.0], (self.length,1) )

		list_of_samples = np.zeros( (self.iterations, self.length,1) )


		#Choose randomly spin to flip
		spin_to_flip = np.random.randint(0,self.length, size=self.iterations)

		for i in range(self.iterations):
			
			new_sample = np.copy(sample)
			new_sample[spin_to_flip[i]] *= -1

			#Compute test (with unnormalized probability bc. of division btw both)
			R = np.exp( self.model.log_prob(new_sample) - self.model.log_prob(sample) )
			#print(R)

			eta = np.random.uniform()
			if R > eta:
				sample[spin_to_flip[i]] *= -1


			# Save the sample
			list_of_samples[i] = np.copy(sample)
			E_loc[i] = self.model.local_energy(self.hamiltonian, sample)


		return list_of_samples, E_loc


