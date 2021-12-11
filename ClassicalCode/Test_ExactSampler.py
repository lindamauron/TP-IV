import numpy as np
import matplotlib.pyplot as plt
import Models
import Samplers

from scipy.optimize import fsolve
###############################################
# Parameters
#Number of particles
n_spins = 15

#Temperature
beta = 1e0 #[eV]

model = Models.MeanField(beta, n_samples=n_spins)

# Comparing Exact sampler and MCMC
exact = Samplers.ExactSampler(model, iterations=5000)
samples_exact = exact.run()

mcmc = Samplers.MCMC(model, iterations=7000)
samples_mem = mcmc.run()
samples_mcmc = samples_mem[2000:]


E_exact = np.zeros(5000)
E_mcmc = np.zeros(5000)
for i in range(5000):
	E_exact[i] = model.energy(samples_exact[i])
	E_mcmc[i] = model.energy(samples_mcmc[i])

plt.figure()
#plt.hist(E_exact, bins='auto')

plt.hist(E_exact, label='Exact sampling', alpha=.8, edgecolor='blue')
plt.hist(E_mcmc, label='Monte-Carlo MC', alpha=.7, edgecolor='yellow')

plt.xlabel('Energy')
plt.ylabel('Occurences')
#plt.title(f"exact")
plt.legend()

'''
plt.figure()
plt.hist(E_mcmc, bins='auto')
plt.xlabel('Energy')
plt.ylabel('Occurences')
plt.title(f"mcmc")
'''

###############################################

'''

engine = Samplers.ExactSampler(model, iterations=1000)
engine.print_infos()
engine.show_distribution()

###############################################
# Loop
samples_memory = engine.run()


print('Loops done')
###############################################
# Treating data


energy_of_sample = np.zeros(samples_memory[:,0].size)
f_loc = np.zeros(samples_memory[:,0].size)
for k in range(energy_of_sample.size):
	energy_of_sample[k] = model.energy(samples_memory[k,:])
	f_loc[k] = model.local_free_energy(samples_memory[k,:])




E_mean = energy_of_sample.mean()

F_ising = model.exact_model.free_energy()
###############################################
# Graphics



plt.figure()
plt.plot(range(energy_of_sample.size), energy_of_sample, 'o')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.title(f"{n_spins} spins, {5000} loops, Mean energy = {E_mean}, beta = {beta}")


#Compute density of states
#E_theoretical, DOS = model.DOS()

plt.figure()
plt.hist(energy_of_sample, bins='auto')
#plt.plot(E_theoretical, n_loops*DOS, 'r-o')
plt.xlabel('Energy')
plt.ylabel('Occurences')
plt.title(f"{n_spins} spins, {5000} loops, beta = {beta}")

plt.figure()
plt.plot(f_loc)
plt.axhline(y=F_ising, color='g', ls=':', label=r'$\mathcal{F}(\{ b_0 \})$')
plt.ylabel(r'$\mathcal{F}$')
plt.xlabel('Step')
plt.title(f"{n_spins} spins, {5000} loops, beta = {beta}")
'''

plt.show();

