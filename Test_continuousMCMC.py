import numpy as np
import matplotlib.pyplot as plt

import MCMC
def probability(sample):
	'''
	Gaussian out-centered and dilated
	'''
	return np.exp(-np.sum((sample-5)**2)/2/3**2)


dim = 10
init = np.random.normal(0, 5, size=dim)



sampler = MCMC.Continuous(std = 1.0)
list_samples, loss = sampler.run(init, probability, 15000)

x = np.linspace(-10,20,1000)

plt.figure()
plt.hist(list_samples[3000:])
plt.plot(x, 12000*np.array([probability(y) for y in x])/np.sqrt(3**2), linestyle=':')


plt.figure()
plt.plot(list_samples)
plt.xlabel('Iteration')
plt.ylabel('Samples')

plt.show()
