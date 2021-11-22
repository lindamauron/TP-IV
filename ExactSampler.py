import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import jit
import itertools 

class ExactSampler:
	'''
	Exact distribution of sample implementation
	Only works for systems with N < 16 samples
	'''

	def __init__(self, model, iterations=5000):
		self.model = model
		self.n_samples = model.n_samples
		self.iterations = iterations
		self.generate_distribution()


	def print_infos(self):
		print('The model is :')
		self.model.print_infos()
		print(f'The number of iterations is {self.iterations}')
		print('-----------------------------------------')

	def generate_distribution(self):
		'''
		Generates the probabilistic distribution
		'''
		s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=self.model.n_samples)))
		prob = []
		for s in s_tuples:
			prob.append( np.exp(-self.model.beta*self.model.energy(s)) )

		self.distribution_samples = np.array(s_tuples)
		self.distribution = np.cumsum(prob)

		# Normalizing
		prob /= self.distribution[-1]
		self.distribution /= self.distribution[-1]


	def show_distribution(self):
		#print(self.distribution)
		plt.figure()
		plt.plot(self.distribution, 'b')
		plt.title(f'Probability distribution, beta = {self.model.beta}, n_spins = {self.n_samples}')
		plt.show();
		
		

	def run(self):

		list_of_samples=[]
		for k in range(self.iterations):
			eta = np.random.uniform()
			diff = [prob for prob in (self.distribution-eta) if prob > 0]

			i = np.argmax(diff)

			list_of_samples.append(self.distribution_samples[i])
		return np.array(list_of_samples)
		
