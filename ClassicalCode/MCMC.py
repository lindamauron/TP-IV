import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit

class MCMC:
	'''Monte-Carlo with Markov Chains implementation'''

	def __init__(self, model, iterations=5000):
		self.model = model
		self.n_samples = model.n_samples
		self.iterations = iterations


	def print_infos(self):
		print('The model is :')
		self.model.print_infos()
		print(f'The number of iterations is {self.iterations}')
		print('-----------------------------------------')



	def run(self, sample=None):
		'''
		Executes the MCMC algorithm
		Input :
		sample (1D array): initial spins of the sample 
							if None, random sample
		Return : list of used samples (2D array)
		'''

		if sample is None:
			sample = np.random.choice([-1.0, 1.0], self.n_samples)

		list_of_samples = np.zeros( (self.iterations, self.n_samples) )
		for i in range(self.iterations):
			#Choose randomly spin to flip
			spin_to_flip = np.random.randint(0,self.n_samples)
			new_sample = np.copy(sample)
			new_sample[spin_to_flip] = -new_sample[spin_to_flip]

			#Compute test (with unnormalized probability bc. of division btw both)
			R = np.exp( self.model.unnormalized_log_probability(new_sample) - self.model.unnormalized_log_probability(sample) )

			eta = np.random.uniform()

			if R > eta:
				sample = np.copy(new_sample)
	
			# Save the sample
			list_of_samples[i,:] = sample

		return list_of_samples
