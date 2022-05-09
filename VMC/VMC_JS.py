import numpy as np
import matplotlib.pyplot as plt
import Models
import DiscreteOperator
import MCMC


L = 20
beta = 1e-2

burning_period = 100

n_variational_loops = 50
learning_rate = 1e-2

model = Models.Jastrow(L)
H = DiscreteOperator.IsingTransverse(L)
engine = MCMC.Quantum(model, H, iterations = 1000)


E = np.zeros( (n_variational_loops,1), dtype=complex )
parameters_memory = np.zeros( (n_variational_loops+1,2,1), dtype=complex )
parameters_memory[0] = model.parameters


for i in range(n_variational_loops):
	print(i)

	if i==0:
		# do MCMC with given parameters for the probability
		samples_memory, E_loc = engine.run()
	samples_memory, E_loc = engine.run( samples_memory[-1] )


	E[i] = E_loc[burning_period:].mean()
	#print(samples_memory[burning_period:])

	# Change parameters descending the gradient
	grad = model.gradient( H, samples_memory[burning_period:] ) 

	# Update with new parametres
	model.parameters = model.parameters-learning_rate*grad
	parameters_memory[i+1] = model.parameters

	if np.isnan(model.parameters).any():
		print('The VMC does not converge, getting to Nan')
		break


np.set_printoptions(precision=3)
print(f'J are {model.parameters}')

E_mean = E.mean()

###################

np.set_printoptions(precision=3)
#print(f'Energy is {E}')



plt.figure()
plt.plot(E.real, 'b', label=r'$E_{\theta}$')
#plt.plot(E.imag, 'r', label='imaginary part')
#plt.plot(np.absolute(E), 'k', label='modulus')
plt.axhline(y=-25.49098969, label=r'$E_0$')
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.legend()
plt.title(f"{model.name} with lr = {learning_rate} and {L} spins, final energy = {E[-1]}")




# Evolution of the parameters
plt.figure()
plt.plot(np.reshape(parameters_memory, (n_variational_loops+1,2)) )
plt.xlabel('Iteration')
plt.ylabel('Parameters')
plt.legend(['J1', 'J2'])
plt.title(f'{model.name} with lr = {learning_rate} and {L} spins')



plt.show();


