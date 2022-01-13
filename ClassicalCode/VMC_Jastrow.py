import numpy as np
import matplotlib.pyplot as plt
import Samplers
import Models


##########################################################
# Parameters
#Number of particles
n_spins = 25

#Temperature
beta = 1e0 #[eV]
 

n_variational_loops = 500

learning_rate = 1e-3

MCMC_iterations=5000
burning_period = 2000
ns = MCMC_iterations-burning_period
n_MCMC = 5


model = Models.Jastrow(beta, n_samples=n_spins, h=1, J=1)
engine = Samplers.MCMC(model, iterations=MCMC_iterations)
engine.print_infos()

print(f'# MCMC is = {MCMC_iterations}, T_therm = {burning_period}, avgg={n_MCMC}, learning_rate={learning_rate}')

###########################################################
# Values of interest

F_lamb = np.zeros(n_variational_loops)
parameters_memory = np.zeros( (n_variational_loops+1, 1) )
parameters_memory[0] = model.parameters

# Variational computation w.r.t. parameters W
print('# Loop, F_lambda, parameters')

for i in range(n_variational_loops):
	list_of_samples = []

	for j in range(n_MCMC):
		# do MCMC with given parameters for the probability
		samples_memory = engine.run()

		list_of_samples = samples_memory[burning_period:]
		for k in range(burning_period, MCMC_iterations):
			F_lamb[i] += model.local_free_energy(samples_memory[k])/(n_MCMC*ns)



	# Change parameters descending the gradient
	grad = model.gradient(list_of_samples) 


	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i+1] = model.parameters

	print(f'{i}, {F_lamb[i]}, {parameters_memory[i+1]}')


np.set_printoptions(precision=3)
print(f'# W is {model.parameters}')
###############################################
# Graphics

F_ising = model.exact_model.free_energy()


# Evolution of the parameters
plt.figure()
plt.plot(range(n_variational_loops+1), parameters_memory)
plt.ylabel('Parameter W')
plt.legend()



plt.figure()
plt.plot(range(n_variational_loops), F_lamb, 'b', label = r'Variational $\mathcal{F}_{\lambda}$')
plt.axhline(y=F_ising, color='g', ls=':', label=r'Analytical $\mathcal{F}$')
plt.ylabel(r'Free Energy $\mathcal{F}$')
plt.legend()




plt.show();