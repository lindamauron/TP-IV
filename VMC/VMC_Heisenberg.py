import numpy as np
import matplotlib.pyplot as plt

import Models
import DiscreteOperator
import MCMC


L = 10
beta = 1e0

burning_period = 500

n_variational_loops = 250
learning_rate = 1e-1

model = Models.Jastrow(L)
H = DiscreteOperator.Heisenberg(L)
engine = MCMC.Quantum(model, H, iterations = 7000)


E = np.zeros( (n_variational_loops,1), dtype=complex )
parameters_memory = np.zeros( (n_variational_loops+1,2,1), dtype=complex )
parameters_memory[0] = model.parameters

for i in range(n_variational_loops):
	print(i)

	# do MCMC with given parameters for the probability
	samples_memory, E_loc = engine.run()
	'''
	plt.figure()
	plt.plot(E_loc)
	plt.title(f'iteration={i}')
	plt.show()
	'''

	E[i] = E_loc[burning_period:].mean()

	# Change parameters descending the gradient
	grad = model.gradient( H, samples_memory[burning_period:] ) 

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i+1] = model.parameters



np.set_printoptions(precision=3)
print(f'Theta is {model.parameters}')

E_mean = E.mean()

###################

np.set_printoptions(precision=3)
#print(f'Energy is {E}')

#-35.61754612
plt.figure()
plt.plot(E.real, 'b', label=r'$E_\lambda$')
#plt.plot(E.imag, 'r', label='imaginary part')
#plt.plot(np.absolute(E), 'k', label='modulus')
plt.axhline(y=-18.06178542, label=r'$E_0$', color='g', ls=':')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.legend()
plt.title(f"{model.name} with lr = {learning_rate} and {L} spins, final energy = {E[-1]}")


plt.figure()
plt.plot( np.reshape(parameters_memory, (n_variational_loops+1,2)) )
plt.xlabel('Iteration')
plt.ylabel(r'Parameters $\theta$')
plt.legend()
plt.title(f'{model.name} with lr = {learning_rate} and {L} spins')


plt.show();


