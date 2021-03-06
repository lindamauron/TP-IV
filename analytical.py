import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg 

N = 20


X = np.array([[0,1], [1,0]], dtype=complex)
Y = np.array([[0,-1j], [1j,0]], dtype=complex)
Z = np.array([[1,0], [0,-1]], dtype=complex)
I = np.array([[1,0], [0,1]], dtype=complex)
'''
print(f'X = {X}, Y={Y}, Z={Z}')

A=sp.kron(Y,Z)

print(A)
'''
XX = sp.kron(X, X)
YY = sp.kron(Y, Y)
ZZ = sp.kron(Z, Z)

'''
# Heisenberg
#H = np.zeros((2**N, 2**N), dtype=complex)
H = sp.kron(sp.kron(X+Y+Z, sp.eye(2**(N-2)) ), X+Y+Z)
#print(H)
for i in range(0,N-1):
	A = sp.kron(sp.kron(sp.eye(2**i), XX + YY + ZZ), sp.eye(2**(N-2-i)) )
	#print(A.shape)
	H += A

#print(H)

print( sp.linalg.eigsh(H, k=2) )
#-35.61754612,
'''

# Ising transverse
H = sp.kron(sp.kron(Z, sp.eye(2**(N-2)) ), Z) + sp.kron( sp.eye(2**(N-1) ), X)
for i in range(0,N-1):
	A = sp.kron(sp.kron(sp.eye(2**i), ZZ), sp.eye(2**(N-2-i)) )

	B = sp.kron(sp.kron(sp.eye(2**i), X), sp.eye(2**(N-1-i)) )
	#print(A.shape)
	H += A + B

print( sp.linalg.eigsh(H, k=2) )


