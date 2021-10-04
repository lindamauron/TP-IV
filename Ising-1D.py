'''
Simulate a chain of N spins s=+/-1 using Metropolis-Hasting Algorithm

0. create vector of N spins pm1

loop
T(x->x') = .....
A = min(a, P(x)/P(x') * T(x->x')/T(x'->x))
0 < eta < 1
if A > eta, x = x'
otherwise x unchanged

'''
import numpy as np



def P_Boltzman(Spins, index):
	''' 
	Compute the Boltzman probability of spin number index
		The energy is given by E_i = exp(-sum_neraestNeighboors S_i*S_j)
	Spins : chain of spins in 1D with periodical boundary conditions
	index : position at which the probability wants to be computed
	'''

	#Computing partition function
	N = Spins.size
	Z = 0
	for i in range(N):
		#Implementing boundary conditions : S_N = S_0
		if i==0:
			Z += np.exp(-Spins[i]*Spins[-1]-Spins[i]*Spins[i+1])
		elif i==N-1:
			Z += np.exp(-Spins[i]*Spins[i-1]-Spins[i]*Spins[0])
		else :
			Z += np.exp(-Spins[i]*Spins[i-1]-Spins[i]*Spins[i+1])

	#Single probability
	if index==0 :
		P_i = np.exp(-Spins[index]*Spins[index+1] - Spins[index]*Spins[-1])
	elif index==N-1:
		P_i = np.exp(-Spins[index]*Spins[0] - Spins[index]*Spins[index-1])
	else :
		P_i = np.exp(-Spins[index]*Spins[index+1] - Spins[index]*Spins[index-1])

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
N = 5

#Number of loops to execute
n_loops = 10

#Creating vector of spins, randomly +1 or -1
S = np.random.choice([-1.0, 1.0], N)
#print(S)

Energy = np.zeros(N)

for i in range(n_loops):
	#Choose randomly spin to flip
	x_new = np.random.randint(0,N)
	S_temp = np.copy(S)
	S_temp[x_new] = -S_temp[x_new]

	#Compute test

	R = P_Boltzman(S_temp, x_new)/P_Boltzman(S, x_new) * T(S_temp, S)/T(S, S_temp)

	eta = np.random.uniform()

	if R > eta:
		S = S_temp




	print(f'S={S}')
