import numpy as np
import matplotlib.pyplot as plt
import Models
import DiscreteOperator
import QMCMC


L = 5
beta = 1e-2

burning_period = 100

n_variational_loops = 1000
learning_rate = 1e-1

model = Models.MeanField(L)
H = DiscreteOperator.IsingTransverse(beta, L, model)
engine = QMCMC.MCMC(model, H, iterations = 1000)

#sample_list, E_loc = engine.run()

E = np.zeros( (n_variational_loops,1), dtype=complex )
parameters_memory = np.zeros( (n_variational_loops,L,1), dtype=complex )


for i in range(n_variational_loops):
	print(i)

	if i==0:
		# do MCMC with given parameters for the probability
		samples_memory, E_loc = engine.run()
	else:
		samples_memory, E_loc = engine.run( samples_memory[-1] )


	E[i] = E_loc.mean()

	# Change parameters descending the gradient
	grad = model.gradient( H, samples_memory[burning_period:] ) 

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i] = model.parameters



np.set_printoptions(precision=3)
print(f'W is {model.parameters}')

E_mean = E.mean()

###################

np.set_printoptions(precision=3)
#print(f'Energy is {E}')

plt.figure()
plt.plot(E, 'o')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.title(f"MF {L} spins, {5000} loops, Mean energy = {E_mean:.4f}, beta = {beta}")


plt.figure()
plt.plot(np.reshape(parameters_memory,(n_variational_loops,L) ) )
plt.xlabel('Iteration')
plt.ylabel('Parameters b_i')
plt.legend( ['b_0', 'b_1', 'b_2', 'b_3', 'b_4', ])
plt.title(f'MF with lr = {learning_rate}, beta = {beta} and {L} spins')


plt.show();


