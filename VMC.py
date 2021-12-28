import numpy as np
import matplotlib.pyplot as plt

import Models
import DiscreteOperator
import QMCMC


L = 10


burning_period = 2000

n_variational_loops = 5
learning_rate = 1e-2

model = Models.MeanField(L)
H = DiscreteOperator.Heisenberg(L,J=-1)
engine = QMCMC.MCMC(model, H, iterations = 7000)

engine.print_infos()
print(f'T_therm = {burning_period}, learning_rate={learning_rate}, n={n_variational_loops}')

E = np.zeros( (n_variational_loops,1), dtype=complex )
parameters_memory = np.zeros( (n_variational_loops+1,model.parameters.size,1), dtype=complex )
parameters_memory[0] = model.parameters

print('-------- Start Variational --------')
for i in range(n_variational_loops):
	print(i)

	# do MCMC with given parameters for the probability
	samples_memory, E_loc = engine.run( )

	'''
	plt.figure()
	plt.plot(E_loc)
	plt.title(i)
	'''

	E[i] = E_loc[burning_period:].mean()

	# Change parameters descending the gradient
	grad = model.gradient( H, samples_memory[burning_period:] ) 

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i+1] = model.parameters



np.set_printoptions(precision=3)
print(f'Theta is {model.parameters} and final energy is {E[-1]}')
print('-------- End Variational --------')


E_mean = E.mean()

###################

np.set_printoptions(precision=3)
print(f'Final energy is {E[-1]}')


'''
# Ising : 
E_10 = 
E_20 = -25.49098969

# Heisenberg
E_10 = 
E_20 = -35.61754612
'''

E_th = H.get_solution()


plt.figure()
plt.plot(E.real, 'b', label=r'$E_\theta$')
plt.axhline(y=E_th, label=r'$E_0$', color='g', ls=':')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.legend()
#plt.title(f"{model.name} with lr = {learning_rate} and {L} spins")


plt.figure()
for i in range(model.parameters.size):
	plt.plot( parameters_memory[:,i,0])
plt.xlabel('Iteration')
plt.ylabel(r'Parameters $\theta$')
#plt.legend([r'$J_1$', r'$J_2$'])
#plt.title(f'{model.name} with lr = {learning_rate} and {L} spins')


plt.show();


