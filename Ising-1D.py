'''
Simulate a chain of N spins s=+/-1 using Metropolis-Hasting Algorithm

'''
import numpy as np
import matplotlib.pyplot as plt


def P_Boltzman(Spins, index):
	''' 
	Compute the Boltzman probability of spin number index
		The energy is given by E_i = -sum_nearestNeighboors S_i*S_j
	Spins : chain of spins in 1D with periodical boundary conditions
	index : position at which the probability wants to be computed
	'''
	N = Spins.size

	#Computing partition function
	Z = 0
	for i in range(N):
		#Implementing boundary condition : S_N = S_0
		if i==N-1:
			Z += np.exp(Spins[i]*Spins[0])
		else :
			Z += np.exp(Spins[i]*Spins[i+1])

	#Single probability
	if index==0 :
		P_i = np.exp(Spins[index]*Spins[index+1] + Spins[index]*Spins[-1])
	elif index==N-1:
		P_i = np.exp(Spins[index]*Spins[0] + Spins[index]*Spins[index-1])
	else :
		P_i = np.exp(Spins[index]*Spins[index+1] + Spins[index]*Spins[index-1])

	return P_i/Z


def T(S_initial, S_next):
	'''
	Computes the local probability to switch spin x_init -> x_next
	S_initial : intial chain of spins in 1D with periodical boundary conditions
	S_next : new spin chain with same conditions
	'''
	return 1.0


###############################################
#Number of particles
N = 1000

#Number of loops to execute
n_loops = 500

#Creating vector of spins, randomly +1 or -1
S = np.random.choice([-1.0, 1.0], N)
#print(S)

Energy = np.zeros(n_loops)

for i in range(n_loops):
	#Choose randomly spin to flip
	x_new = np.random.randint(0,N)
	S_new = np.copy(S)
	S_new[x_new] = -S_new[x_new]

	#Compute test
	R = P_Boltzman(S_new, x_new)/P_Boltzman(S, x_new)

	eta = np.random.uniform()

	if R > eta:
		S = np.copy(S_new)

	
	#Computing the enrgy of the system
	for k in range(N):
		if k != N-1:
			Energy[i] -= S[k]*S[k+1]
		else :
			Energy[i] -= S[k]*S[0]


#	if i==0 or i==n_loops-1:
#		print(f'S={S} and E = {Energy[i]}')

#E_moy = np.sum(Energy)/n_loops

'''
plt.plot(Energy, 'o')
plt.title(f"N_particles = {N}, N_loops = {n_loops}, E_mean = {E_moy}")
plt.xlabel('Iteration')
plt.ylabel("Energy")
plt.show();
'''

plt.hist(Energy, bins='auto')
plt.ylabel('Occurences')
plt.show();
