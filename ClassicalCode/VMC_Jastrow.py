import numpy as np
import matplotlib.pyplot as plt
import MCMC
import Jastrow as JS
import ExactIsing1D


##########################################################
# Parameters
#Number of particles
n_spins = 5

#Temperature
beta = 1e0 #[eV]

n_variational_loops = 1000

learning_rate = 1e-2


size_of_mean = 1000
MCMC_iterations = 5000
burning_period = 2000

model = JS.Jastrow(beta, n_samples=n_spins)

engine = MCMC.MCMC(model, iterations=MCMC_iterations)
engine.print_infos()


###########################################################
# Values of interest

F_lamb = np.zeros(n_variational_loops)
parameters_memory = np.zeros( (n_variational_loops, model.n_samples, model.n_samples) )


# Variational computation w.r.t. parameters {W_ij}

for i in range(n_variational_loops):
	list_of_samples = []
	print(i)

	if i ==0:
		# do MCMC with given parameters for the probability
		samples_memory = engine.run()
	else:
		samples_memory = engine.run( samples_memory[-1,:] )


	ns = np.random.choice(range(burning_period,MCMC_iterations), size=size_of_mean, replace=False)
	for k in ns:
		F_lamb[i] += model.local_free_energy(samples_memory[k,:])/len(ns)
		list_of_samples.append(samples_memory[k,:])

	list_of_samples = np.array(list_of_samples)

	# Change parameters descending the gradient
	grad = model.gradient(list_of_samples) 

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i,:,:] = model.parameters



np.set_printoptions(precision=3)
print(f'W is {model.parameters}')
###############################################
# Graphics
F_ising = model.exact_model.free_energy()

# Evolution of the parameters
plt.figure()
for k in range(model.n_samples):
	for l in range(model.n_samples):
		if k < l:
			plt.plot(range(n_variational_loops), parameters_memory[:,k,l], label=f'W_{k},{l}')
plt.xlabel('Iteration')
plt.ylabel('Parameters W_ij')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title(f'JS with lr = {learning_rate}, beta = {beta} and {n_spins} spins')
#plt.savefig(f"VMC_Jastrow_parameters(lr={learning_rate}).png", bbox_inches="tight")



plt.figure()
#plt.plot(range(n_variational_loops+1), F_loc, 'b', ls=':', label = r'$\mathcal{F}_{loc}(\{ b_i \})$')
plt.plot(range(n_variational_loops), F_lamb, 'b', label = r'$\mathcal{F}_{\lambda}(\{ b_i \})$')
plt.axhline(y=F_ising, color='g', ls=':', label=r'$\mathcal{F}(\{ b_0 \})$')
plt.ylabel('Free Energy')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.title(f'JS with lr = {learning_rate}, beta = {beta} and {n_spins} spins')
#plt.savefig(f"VMC_Jastrow_freeEnergy(lr={learning_rate}).png", bbox_inches="tight")



plt.show();