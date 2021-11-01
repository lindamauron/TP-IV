import numpy as np
import matplotlib.pyplot as plt
import MCMC
import MeanField as MF
import ExactIsing1D

from scipy.optimize import fsolve

##########################################################
# Parameters
#Number of particles
n_spins = 25

#Temperature
beta = 1e0 #[eV]

n_variational_loops = 2000

learning_rate = 1e-1
size_of_mean = 500


model = MF.MeanField(beta, n_samples=n_spins)
exact_model = ExactIsing1D.ExactIsing1D(beta, n_samples=n_spins)

engine = MCMC.MCMC(model, warm_up_iterations=5000)
engine.print_infos()


###########################################################
# Values of interest

F_loc = np.zeros(n_variational_loops+1)
F_lamb = np.zeros(n_variational_loops+1)
parameters_memory = np.zeros( (n_variational_loops, model.n_samples) )


# Variational computation w.r.t. parameters {b_i}

for i in range(n_variational_loops):
	print(i)
	if i == 0:
		#Creating vector of spins, randomly +1 or -1
		sample = np.random.choice([-1.0, 1.0], model.n_samples)

		#Warm-up iterations
		samples_memory = engine.run(sample, flag_warm_up=True)
		sample = samples_memory[-1,:]

		F_loc[0] = model.local_free_energy(sample)

		ns = range(4000, 5000)
		#ns = np.random.choice(range(5000), size=size_of_mean, replace=False)
		for k in ns:
			F_lamb[0] += model.local_free_energy(samples_memory[k,:])
		F_lamb[0] /= len(ns)



	# do MCMC with given parameters for the probability
	list_of_samples = engine.run(sample)
	sample = samples_memory[-1,:]

	F_loc[i+1] = model.local_free_energy(sample)

	ns = np.random.choice(range(1000), size=size_of_mean, replace=False)
	for k in ns:
		F_lamb[i+1] += model.local_free_energy(samples_memory[k,:])
		np.append(list_of_samples, samples_memory[k,:])
	F_lamb[i+1] /= ns.size

	# Change parameters descending the gradient
	grad = model.gradient(list_of_samples) 

	# Check that gradient in non-zero
	if np.linalg.norm(grad) < 1e-30:
		break

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i,:] = model.parameters




#print(f'b is {model.parameters}')
###############################################
# Graphics
J=1
h=1
fct = lambda b_ana : b_ana - h - 2*J*np.tanh(beta*b_ana)
b_sol = fsolve(fct, 2.9)
F_lamb_exact = -model.n_samples*( np.log( 2*np.cosh(model.beta*b_sol)) )/model.beta

F_ising = exact_model.free_energy()


# Evolutiono of the parameters
plt.figure()
plt.plot(range(n_variational_loops), parameters_memory)
plt.axhline(y=b_sol, color='k', ls=':', label=r'Theoretical solution $b_0$')
plt.xlabel('Iteration')
plt.ylabel('Parameters b_i')
plt.legend()
plt.title(f'Variational with lr = {learning_rate}, beta = {beta} and {n_spins} spins')



plt.figure()
plt.plot(range(n_variational_loops+1), F_loc, 'b', label = r'$\mathcal{F}_{loc}(\{ b_i \})$')
plt.plot(range(n_variational_loops+1), F_lamb, 'b', label = r'$\mathcal{F}_{\lambda}(\{ b_i \})$')
plt.axhline(y=F_lamb_exact, color='r', ls=':', label=r'$\mathcal{F}_{\lambda}(\{ b_0 \})$')
plt.axhline(y=F_ising, color='g', ls=':', label=r'$\mathcal{F}(\{ b_0 \})$')
plt.ylabel('Free Energy')
plt.legend()
plt.title(f'Variational with lr = {learning_rate}, beta = {beta} and {n_spins} spins')




plt.show();