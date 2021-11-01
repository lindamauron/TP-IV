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

test.update_parameters(-test.parameters*3.0)
print(f'Mean field energy should be {90} and is {test.energy(sample)}')

print(test.GD(1e-3))

print('-------------------------------')
