import numpy as np
import matplotlib.pyplot as plt

n = 50

J=1
h=1
I = np.array([[1,0],[0,1]])
phi = np.linspace(0, 2*np.pi, n)
theta = np.linspace(0, np.pi, n)


#print('# theta , phi, Emin')
for t2 in np.linspace(0, 1, 11)*np.pi:
	for phi2 in np.linspace(0, 2, 21)*np.pi: #[1/2*np.pi]:


		E0 = np.zeros( (n,n) )
		for k in range(n):
			phi1 = phi[k]
			for j in range(n):
				t1 = theta[j]
				c1 = np.cos(t1)
				s1 = np.sin(t1)
				c2 = np.cos(t2)
				s2 = np.sin(t2)

				X = np.array([[np.cos(phi1)*s1, -np.exp(-1j*phi1)*(c1*np.cos(phi1) + 1j*np.sin(phi1))], 
					[-np.exp(1j*phi1)*(c1*np.cos(phi1) - 1j*np.sin(phi1)), -np.cos(phi1)*s1]])
				Z1 = np.array([[c1, np.exp(-1j*phi1)*s1], [np.exp(1j*phi1)*s1, -c1]])
				Z2 = np.array([[c2, np.exp(-1j*phi2)*s2], [np.exp(1j*phi2)*s2, -c2]])



				H = np.array(
					[[J*c1*c2 + h*np.cos(phi1)*s1, np.exp(-1j*phi2)*J*c1*s2, -np.exp(-1j*phi1)*h*(np.cos(phi1)*c1+1j*np.sin(phi1)) + np.exp(-1j*phi1)*J*c2*s1, np.exp(-1j*phi1-1j*phi2)*J*s1*s2], 
					[np.exp(1j*phi2)*J*c1*s2, -J*c1*c2+h*np.cos(phi1)*s1, np.exp(-1j*phi1 + 1j*phi2)*J*s1*s2, -np.exp(-1j*phi1)*h*(np.cos(phi1)*c1 + 1j*np.sin(phi1)) - np.exp(-1j*phi1)*J*c2*s1],
					[-np.exp(1j*phi1)*h*(c1*np.cos(phi1)-1j*np.sin(phi1)) + np.exp(1j*phi1)*J*c2*s1, np.exp(1j*phi1-1j*phi2)*J*s1*s2, -J*c1*c2-h*np.cos(phi1)*s1, -np.exp(-1j*phi2)*J*c1*s2],
					[np.exp(1j*phi1+1j*phi2)*J*s1*s2, -np.exp(1j*phi1)*h*(np.cos(phi1)*c1-1j*np.sin(phi1))-np.exp(1j*phi1)*J*c2*s1, -np.exp(1j*phi2)*J*c1*s2, J*c1*c2-h*np.cos(phi1)*s1]]
					, dtype=complex)

				if not np.allclose(H, J*np.kron(Z1, Z2) + h*np.kron(X, I)):
					print('ARGHHHHHHHHHHHHHHHHH')



plt.show();
print('# -------------------------------------------------------------')