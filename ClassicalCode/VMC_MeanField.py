import numpy as np
import matplotlib.pyplot as plt
import Samplers
import Models

from scipy.optimize import fsolve

##########################################################
# Parameters
#Number of particles
n_spins = 25

#Temperature
beta = 1e0 #[eV]
 

n_variational_loops = 1

learning_rate = 1e-1

MCMC_iterations=5000
burning_period = 2000
ns = MCMC_iterations-burning_period
n_MCMC = 5

J=1
h=1
model = Models.MeanField(beta, n_samples=n_spins, h=h, J=J)
engine = Samplers.MCMC(model, iterations=MCMC_iterations)
engine.print_infos()

print(f'# MCMC is = {MCMC_iterations}, T_therm = {burning_period}, avgg={n_MCMC}, learning_rate={learning_rate}')

###########################################################
# Values of interest

F_lamb = np.zeros(n_variational_loops)
parameters_memory = np.zeros( (n_variational_loops+1, model.parameters.size) )
parameters_memory[0] = model.parameters

# Variational computation w.r.t. parameters {b_i}
print('# Loop, F_lambda, parameters')

for i in range(n_variational_loops):
	list_of_samples = []
	#print(f'# Loop {i}')

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

	print(f'{i}, {F_lamb[i]}, {parameters_memory[i+1,:]}')


np.set_printoptions(precision=3)
print(f'# b is {model.parameters}')


###############################################
# Graphics
fct = lambda b_ana : b_ana - h - 2*J*np.tanh(beta*b_ana)
b_sol = fsolve(fct, 2.9)


F_ising = model.exact_model.free_energy()


# Evolution of the parameters
plt.figure()
plt.plot(range(n_variational_loops+1), parameters_memory)
plt.axhline(y=b_sol, color='g', ls=':', label=r'Analytical solution $b$')
plt.ylabel('Parameters b_i')
plt.legend()
#plt.title(f'MF with lr = {learning_rate}, beta = {beta} and {n_spins} spins')



plt.figure()
plt.plot(range(n_variational_loops), F_lamb, 'b', label = r'Variational $\mathcal{F}_{\lambda}$')
plt.axhline(y=F_ising, color='g', ls=':', label=r'Analytical $\mathcal{F}$')
plt.ylabel(r'Free Energy $\mathcal{F}$')
plt.legend()
#plt.title(f'MF with lr = {learning_rate}, beta = {beta} and {n_spins} spins')




plt.show();