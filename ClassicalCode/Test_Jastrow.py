import numpy as np
import matplotlib.pyplot as plt
import Models

import itertools

# Testing creation of the classes
default_class = Models.Jastrow(beta=0.01, n_samples = 10)
print(f'Default class :')
default_class.print_infos()
print('-------------------------------')

no_interaction = Models.Jastrow(beta=0.01, n_samples=15, J = 0)
print('Class without interactions :')
no_interaction.print_infos()
print('-------------------------------')

no_field = Models.Jastrow(beta=0.01, n_samples=10, h = 0)
print('Class without field :')
no_field.print_infos()
print('-------------------------------')




# Testing partition function
beta = 1e0
for n_spins in [3,5,7,10,12,15]:
	test_partition = Models.Jastrow_2(beta=beta, n_samples=n_spins)
	test_partition.parameters *= 1
	s_tuples = np.array(list(k for k in itertools.product( [1.0, -1.0], repeat=n_spins)))
	#print(s_tuples)
	Z = 0
	for s in s_tuples:
		Z += np.exp(-beta*test_partition.energy(s))

	print(f'n={n_spins}, real Z={Z} and implemented Z={test_partition.partition_function()}')




# Testing gradient
test_grad = Models.Jastrow_test(beta=1e-2, n_samples=3)
test_grad.parameters = test_grad.parameters
s = np.random.choice([-1.0, 1.0], test_grad.n_samples)
#s = np.ones(test_grad.n_samples)

df_dx = 0
for x0 in [0.001, 0.0001]:
	f = test_grad.local_free_energy(s)
	
	test_grad.parameters = test_grad.parameters + x0

	f_plus = test_grad.local_free_energy(s)

	df_dx = (f_plus - f)/x0

	test_grad.parameters = test_grad.parameters - x0

	np.set_printoptions(precision=3)
	print(f'finite difference : {df_dx} \n derivative : {test_grad.local_gradient(s)}')
