import numpy as np
import matplotlib.pyplot as plt
import Models


length = 3
beta = 10

Iclass = Models.Ising(beta, length)
Iclass.print_infos()

sample = np.random.choice([-1.0, 1.0], (length, 1) )

print(sample)

print(f'energy = {Iclass.energy(sample)} ')