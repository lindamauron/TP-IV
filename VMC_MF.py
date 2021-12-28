import numpy as np
import matplotlib.pyplot as plt

import Models
import DiscreteOperator
import QMCMC


L = 20
beta = 1e-2

burning_period = 500

n_variational_loops = 50
learning_rate = 1e-1

model = Models.MeanField(L)
H = DiscreteOperator.IsingTransverse(L)
engine = QMCMC.MCMC(model, H, iterations = 7000)


E = np.zeros( (n_variational_loops,1), dtype=complex )
parameters_memory = np.zeros( (n_variational_loops+1,1), dtype=complex )
parameters_memory[0] = model.parameters

for i in range(n_variational_loops):
	print(i)

	# do MCMC with given parameters for the probability
	samples_memory, E_loc = engine.run()


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

plt.figure()
plt.plot(E.real, 'b', label='real part')
#plt.plot(E.imag, 'r', label='imaginary part')
#plt.plot(np.absolute(E), 'k', label='modulus')
plt.axhline(y=-25.49098969, label=r'$E_0$')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.legend()
plt.title(f"{model.name} with lr = {learning_rate} and {L} spins, final energy = {E[-1]}")


#minE = -25.49098969
plt.figure()
plt.plot( parameters_memory )
plt.xlabel('Iteration')
plt.ylabel(r'Parameters $\theta$')
plt.legend()
plt.title(f'{model.name} with lr = {learning_rate} and {L} spins')


plt.show();


