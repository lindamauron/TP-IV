import numpy as np
import matplotlib.pyplot as plt
import Models
import DiscreteOperator
import MCMC
import time

L = 20
beta = 1e-2

burning_period = 100

n_variational_loops = 5000
learning_rate = 1e-1

'''
model = Models.MeanField(L)
H = DiscreteOperator.IsingTransverse(beta, L)
engine = MCMC.Quantum(model, H, iterations = 1000)


engine.print_infos()
'''


'''
# Testing computation of energy
s = np.ones( (L,1) )

E_, s_prime, n_conns = H.get_H_terms(s)
print(s_prime)
# print(E) #first is L*J, then only h
'''

'''
v = np.array([0,1,2,3])
print(f'e^x = {np.exp(v)}')
print(f'1+e^x = {1+np.exp(v)}')
print(f'1/(1+e^x) = {1/(1+np.exp(v))}')
print(f'sum x_i/(1+e^x_i) = {v.T @ (1/(1+np.exp(v)))}')
'''


'''
# Testing derivative of energy
#s_list, E_loc = engine.run()
s = np.random.choice([-1.0, 1.0], (L,1) )
E = model.local_energy(H,s)

model.parameters += 1e-3
Ep = model.local_energy(H,s)
model.parameters -= 1e-3


D = 0.5*s.T @ (1/(1+np.exp(model.parameters*s)) )
der = (model.local_energy(H,s) - E)*D

print(f'Real derivative is {der} and difference = {(Ep-E)*1e3}')
'''

'''
# Testing MCMC conditions
plt.figure()
plt.plot(E_loc)
plt.xlabel('iteration')
plt.ylabel('Energy')
'''

'''
###################
# Testing analytical solution
H = DiscreteOperator.Heisenberg(J=-1,length=10)


np.set_printoptions(precision=3)
print(H.get_solution())
'''



plt.show();


