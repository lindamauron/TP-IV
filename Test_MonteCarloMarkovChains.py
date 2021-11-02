'''
Simulate a chain of N spins s=+/-1 using MCMC with Metropolis-Hasting Algorithm
'''
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import ExactIsing1D
import MCMC
###############################################
# Functions 


###############################################
# Parameters
#Number of particles
n_spins = 100

#Inverse emperature
beta = 1e-2 #[1/eV]

#Warm up
warm_up_iteration = 5000

#Number of loops to execute
n_loops = 50000

model = ExactIsing1D.ExactIsing1D(beta, n_spins, type_of_h='zero')
engine = MCMC.MCMC(model)

engine.print_infos()

###############################################

#Creating vector of spins, randomly +1 or -1
sample = np.random.choice([-1.0, 1.0], model.n_samples)
#print(f'{sample}, {energy(sample)}')

# Warm up
sample, _ = engine.run(sample, flag_warm_up=True)

# MCMC loop
sample, energy_of_sample = engine.run(sample, iterations=n_loops)


print('Loops done')
###############################################
# Graphics
E_mean = np.sum(energy_of_sample)/n_loops

plt.figure()
plt.plot(range(energy_of_sample.size), energy_of_sample, 'o')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.title(f"{n_spins} spins, {n_loops} loops, Mean energy = {E_mean}, beta = {beta}")


#Compute density of states
E_theoretical, DOS = model.DOS()

plt.figure()
plt.hist(energy_of_sample, bins='auto')
plt.plot(E_theoretical, n_loops*DOS, 'r-o')
plt.xlabel('Energy')
plt.ylabel('Occurences')
plt.title(f"{n_spins} spins, {n_loops} loops, beta = {beta}")
plt.show();


