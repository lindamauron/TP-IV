import numpy as np
import matplotlib.pyplot as plt
import Jastrow as JS

import itertools
'''
# Testing creation of the classes
default_class = JS.Jastrow(beta=0.01, n_samples = 10)
print(f'Default class :')
default_class.print_infos()
print('-------------------------------')

no_interaction = JS.Jastrow(beta=0.01, n_samples=15, type_of_J = 'zero')
print('Class without interactions :')
no_interaction.print_infos()
print('-------------------------------')

no_field = JS.Jastrow(beta=0.01, n_samples=10, type_of_h = 'zero')
print('Class without field :')
no_field.print_infos()
print('-------------------------------')

# Testing different probabilities
sample = np.ones(30)
test = JS.Jastrow(beta=1, n_samples=30)

# MF energy = -sum b_k s_k with b_k = 1 by default => e = 0.5N(N+1)
print(f'Jastrow energy should be {0.5*30*29} and is {test.energy(sample)}')

test.parameters = -test.parameters
sample = 3.0*sample
print(f'Jastrow energy should be {-3**2*0.5*30*29} and is {test.energy(sample)}')


print('-------------------------------')

# Testing partition function
beta = 1e0
for n_spins in [3,5,7,10,12,15]:
	test_partition = JS.Jastrow(beta=beta, n_samples=n_spins)
	test_partition.parameters = -5*test_partition.parameters
	s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=n_spins)))
	#print(s_tuples)
	Z = 0
	for s in s_tuples:
		Z += np.exp(-beta*test_partition.energy(s))

	print(f'n={n_spins}, real Z={Z} and implemented Z={test_partition.partition_function()}')
'''

# Testing gradient
test_grad = JS.Jastrow(beta=1e-2, n_samples=3)
test_grad.parameters = -0.1*test_grad.parameters
s = np.random.choice([-1.0, 1.0], test_grad.n_samples)
#s = np.ones(test_grad.n_samples)

df_dx = np.zeros( (test_grad.n_samples, test_grad.n_samples) )
for x0 in [0.001, 0.0001]:
	f = test_grad.local_free_energy(s)
	for k in range(test_grad.n_samples):
		for l in range(test_grad.n_samples):
			test_grad.parameters[k,l] = test_grad.parameters[k,l] + x0

			f_plus = test_grad.local_free_energy(s)

			df_dx[k,l] = (f_plus - f)/x0

			test_grad.parameters[k,l] = test_grad.parameters[k,l] - x0

	np.set_printoptions(precision=3)
	print(f'la diff finie vaut {df_dx} \n et la dérivée {test_grad.local_gradient(s)}')

