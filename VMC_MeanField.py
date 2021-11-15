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
 
burning_period = 1000

n_variational_loops = 10000

learning_rate = 1e-3

number_MCMC = 1
MCMC_iterations=5000
size_of_mean = 1000

model = MF.MeanField(beta, n_samples=n_spins)

engine = MCMC.MCMC(model, iterations=MCMC_iterations)
engine.print_infos()


###########################################################
# Values of interest

F_lamb = np.zeros(n_variational_loops)
parameters_memory = np.zeros( (n_variational_loops, model.n_samples) )


# Variational computation w.r.t. parameters {b_i}

for i in range(n_variational_loops):
	list_of_samples = []
	print(i)

	for j in range(number_MCMC):
		# do MCMC with given parameters for the probability
		samples_memory = engine.run()


		ns = np.random.choice(range(burning_period,MCMC_iterations), size=size_of_mean, replace=False)
		for k in ns:
			F_lamb[i] += model.local_free_energy(samples_memory[k,:])/len(ns)
			list_of_samples.append(samples_memory[k,:])

	F_lamb[i] /= number_MCMC

	list_of_samples = np.array(list_of_samples)

	# Change parameters descending the gradient
	grad = model.gradient(list_of_samples) 


	# Check that gradient in non-zero
	#if np.linalg.norm(grad) < 1e-30:
	#	break

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i,:] = model.parameters



np.set_printoptions(precision=3)
print(f'b is {model.parameters}')
###############################################
# Graphics
J=1
h=1
fct = lambda b_ana : b_ana - h - 2*J*np.tanh(beta*b_ana)
b_sol = fsolve(fct, 2.9)
#F_lamb_exact = -model.n_samples*np.log( 2*np.cosh(model.beta*b_sol) )/model.beta

F_ising = model.exact_model.free_energy()


# Evolution of the parameters
plt.figure()
plt.plot(range(n_variational_loops), parameters_memory)
plt.axhline(y=b_sol, color='k', ls=':', label=r'Theoretical solution $b_0$')
plt.xlabel('Iteration')
plt.ylabel('Parameters b_i')
plt.legend()
plt.title(f'MF with lr = {learning_rate}, beta = {beta} and {n_spins} spins')



plt.figure()
#plt.plot(range(n_variational_loops+1), F_loc, 'b', ls=':', label = r'$\mathcal{F}_{loc}(\{ b_i \})$')
plt.plot(range(n_variational_loops), F_lamb, 'b', label = r'$\mathcal{F}_{\lambda}(\{ b_i \})$')
#plt.axhline(y=F_lamb_exact, color='r', ls=':', label=r'$\mathcal{F}_{\lambda}(\{ b_0 \})$')
plt.axhline(y=F_ising, color='g', ls=':', label=r'$\mathcal{F}$')
plt.ylabel('Free Energy')
plt.legend()
plt.title(f'MF with lr = {learning_rate}, beta = {beta} and {n_spins} spins')




plt.show();