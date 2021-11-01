import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit

class MCMC:
	'''Monte-Carlo with Markov Chains implementation'''

	def __init__(self, model, warm_up_iterations=5000):
		self.model = model
		self.beta = model.beta
		self.n_samples = model.n_samples
		self.warm_up_iterations = warm_up_iterations


	def print_infos(self):
		print('The model is :')
		self.model.print_infos()
		print(f'The warm up {self.warm_up_iterations}')
		print('-----------------------------------------')



	def run(self, sample, iterations=1000, flag_warm_up=False):
		'''
		Executes the MCMC algorithm
		Input :
		sample (1D array): initial spins of the sample 
		iterations (int): number of loops to execute
		flag_warm_up (bool): if True, executes the warm up with warm_up_iterations loops

		Return : list of used samples (2D array)
		'''

		if flag_warm_up:
			iterations = self.warm_up_iterations

		list_of_samples = np.zeros( (iterations, self.model.n_samples) )
		for i in range(iterations):
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
