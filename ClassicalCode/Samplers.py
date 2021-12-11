import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import itertools 


class MCMC:
	'''Monte-Carlo with Markov Chains implementation'''

	def __init__(self, model, iterations=5000):
		self.model = model
		self.n_samples = model.n_samples
		self.iterations = iterations


	def print_infos(self):
		print('# The model is :')
		self.model.print_infos()
		print(f'# The number of iterations is {self.iterations}')
		print('############################################')



	def run(self, sample=None):
		'''
		Executes the MCMC algorithm
		Input :
		sample (1D array): initial spins of the sample 
							if None, random sample
		Return : list of used samples (2D array)
		'''

		if sample is None:
			sample = np.random.choice([-1.0, 1.0], (self.n_samples,1))

		list_of_samples = np.zeros( (self.iterations, self.n_samples,1) )
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
		print('# The model is :')
		self.model.print_infos()
		print(f'# The number of iterations is {self.iterations}')
		print('############################################')

	def generate_distribution(self):
		'''
		Generates the probabilistic distribution
		'''
		s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=self.model.n_samples)))
		s_tuples = np.reshape( s_tuples, (2**self.n_samples, self.n_samples,1) )
		prob = []
		for s in s_tuples:
			prob.append( np.exp(self.model.beta*self.model.energy(s)) )

		self.distribution_samples = np.array(s_tuples)
		self.distribution = np.cumsum(prob)

		# Normalizing
		prob /= self.distribution[-1]
		self.distribution /= self.distribution[-1]

		return


	def show_distribution(self):
		#print(self.distribution)
		plt.figure()
		plt.plot(self.distribution, 'b')
		plt.title(f'Probability distribution, beta = {self.model.beta}, n_spins = {self.n_samples}')
		plt.show();

		return
		
		

	def run(self):

		list_of_samples=[]
		for k in range(self.iterations):
			eta = np.random.uniform()
			diff = [prob for prob in (self.distribution-eta) if prob > 0]

			i = np.argmax(diff)

			list_of_samples.append(self.distribution_samples[i])

		return np.array(list_of_samples)
		
