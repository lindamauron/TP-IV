import numpy as np
import matplotlib.pyplot as plt
import MeanField as MF

import itertools

# Testing creation of the class
default_class = MF.MeanField(beta=0.01, n_samples = 10)
#print(f'Default class :')
#default_class.print_infos()


no_interaction = MF.MeanField(beta=0.01, n_samples=15, type_of_J = 'zero')
#no_interaction.print_infos()

no_field = MF.MeanField(beta=0.01, n_samples=10, type_of_h = 'zero')
#no_field.print_infos()


# Testing different probabilities
sample = np.ones(30)
test = MF.MeanField(beta=1, n_samples=30)

# MF energy = -sum b_k s_k with b_k = 1 by default
print(f'Mean field energy should be {-30} and is {test.energy(sample)}')

test.parameters = -test.parameters*3.0
print(f'Mean field energy should be {90} and is {test.energy(sample)}')


print('-------------------------------')
print('-------------------------------')

# Testing partition function
beta = 1e-2
for n_spins in [3,5,7,10,12,15]:
	test_partition = MF.MeanField(beta=beta, n_samples=n_spins)
	s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=n_spins)))
	#print(s_tuples)
	Z = 0
	for s in s_tuples:
		Z += np.exp(-beta*test_partition.energy(s))

	print(f'n={n_spins}, real Z={Z} and implemented Z={test_partition.partition_function()}')


# Testing gradient
test_grad = MF.MeanField(beta=10, n_samples=5)
s = np.random.choice([-1.0, 1.0], test_grad.n_samples)
#s = np.ones(test_grad.n_samples)

df_dx = np.zeros( test_grad.n_samples)
for x0 in [0.1, 0.01, 0.001, 0.0001]:
	f = test_grad.log_probability(s)
	for k in range(test_grad.n_samples):
		test_grad.parameters[k] = test_grad.parameters[k] + x0

		f_plus = test_grad.log_probability(s)

		df_dx[k] = (f_plus - f)/x0

		test_grad.parameters[k] = test_grad.parameters[k] - x0

	print(f'la diff finie vaut {df_dx} \n et la dérivée {test_grad.local_gradient(s)}')

