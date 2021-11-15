import numpy as np
import matplotlib.pyplot as plt
import MCMC
import Jastrow as JS
import MeanField as MF

from scipy.optimize import fsolve
###############################################
# Parameters
#Number of particles
n_spins = 25

#Temperature
beta = 1e0 #[eV]

model = MF.MeanField(beta, n_samples=n_spins)

'''
J=1
h=1
fct = lambda b_ana : b_ana - h - 2*J*np.tanh(beta*b_ana)
b_sol = fsolve(fct, 2.9)
model.parameters = b_sol*np.ones(n_spins)
'''

engine = MCMC.MCMC(model, iterations=5000)
engine.print_infos()
###############################################

#Creating vector of spins, randomly +1 or -1
sample = np.random.choice([-1.0, 1.0], model.n_samples)

# Warm up
samples_memory = engine.run( )
sample = samples_memory[-1,:]

# MCMC loop
#samples = engine.run(sample)


print('Loops done')
###############################################
# Treating data


energy_of_sample = np.zeros(samples_memory[:,0].size)
f_loc = np.zeros(samples_memory[:,0].size)
for k in range(energy_of_sample.size):
	energy_of_sample[k] = model.energy(samples_memory[k,:])
	f_loc[k] = model.local_free_energy(samples_memory[k,:])




E_mean = np.sum(energy_of_sample)/5000

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


plt.show();


