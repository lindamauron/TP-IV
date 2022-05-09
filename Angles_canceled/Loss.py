import numpy as np
import matplotlib.pyplot as plt

n = 20
pi = np.pi

J=1
h=1

t1p = np.reshape(np.linspace(0, pi, n), (n,1))
t2p = np.reshape(np.linspace(0, pi, n), (n,1))

xx,yy=np.meshgrid(t1p, t2p)


p1 = np.random.uniform(0,2)*pi
p2 = np.random.uniform(0,2)*pi


p1p = np.random.uniform(0,2)*pi
p2p = np.random.uniform(0,2)*pi


for t2 in np.linspace(0, 1, 11)*pi : #[np.random.uniform(0,1)*pi]:
	for t1 in np.linspace(0, 1, 11)*pi:

		E = np.zeros( (n,n), dtype=complex )
		E += J*(np.cos(t1/2)*np.cos(t1p/2) - np.exp(1j*(p1p-p1))*np.sin(t1/2)*np.sin(t1p/2)) * (np.cos(t2/2)*np.cos(t2p/2) - np.exp(1j*(p2p-p2))*np.sin(t2/2)*np.sin(t2p/2)).T
		# X_term Ising
		E += h*(np.exp(1j*p1p)*np.cos(t1/2)*np.sin(t1p/2) + np.exp(-1j*p1)*np.sin(t1/2)*np.cos(t1p/2)) * (np.cos(t2/2)*np.cos(t2p/2) + np.exp(1j*(p2p-p2))*np.sin(t2/2)*np.sin(t2p/2)).T

		# Y,X terms Heisenberg
		#E += -J*(np.exp(-1j*p1)*np.sin(t1/2)*np.cos(t1p/2)-np.exp(1j*p1p)*np.cos(t1/2)*np.sin(t1p/2)) * (np.exp(-1j*p2)*np.sin(t2/2)*np.cos(t2p/2)-np.exp(1j*p2p)*np.cos(t2/2)*np.sin(t2p/2)).T
		#E += J*(np.exp(1j*p1p)*np.cos(t1/2)*np.sin(t1p/2) + np.exp(-1j*p1)*np.sin(t1/2)*np.cos(t1p/2)) * (np.exp(1j*p2p)*np.cos(t2/2)*np.sin(t2p/2) + np.exp(-1j*p2)*np.sin(t2/2)*np.cos(t2p/2)).T

		fig = plt.figure()
		fig.suptitle("$\\theta_1 = %.2f \pi$,$\phi_1 = %.2f \pi$ \n $\\theta_2 = %.2f \pi$, $ \phi_2 = %.2f \pi$ \n $\phi_1' = %.2f \pi, \phi_2' = %.2f \pi$" % (t1/pi, p1/pi, t2/pi, p2/pi, p1p/pi, p2p/pi), fontsize=16)

		ax = fig.add_subplot(121, projection='3d')
		ax.scatter(xx, yy, np.real(E), zdir='E', c=np.real(E))
		ax.set_xlabel(r"$\theta'_1$")
		ax.set_ylabel(r"$\theta'_2$")
		ax.set_zlabel(r'$\langle \mathcal{H} \rangle$')
		plt.xticks(ticks=[0, pi/4, pi/2, 3*pi/4, pi], labels=[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
		plt.yticks(ticks=[0, pi/4, pi/2, 3*pi/4, pi], labels=[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
		#plt.yticks(ticks=[0, pi/2, pi, 3*pi/2, 2*pi], labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
		plt.title(r'Real part' )

		ax = fig.add_subplot(122, projection='3d')
		ax.scatter(xx, yy, np.imag(E), zdir='E', c=np.imag(E))
		ax.set_xlabel(r"$\theta'_1$")
		ax.set_ylabel(r"$\theta'_2$")
		ax.set_zlabel(r'$\langle \mathcal{H} \rangle$')
		plt.xticks(ticks=[0, pi/4, pi/2, 3*pi/4, pi], labels=[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
		plt.yticks(ticks=[0, pi/4, pi/2, 3*pi/4, pi], labels=[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
		#plt.yticks(ticks=[0, pi/2, pi, 3*pi/2, 2*pi], labels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
		plt.title(r'Imaginary part' )


plt.show();
print('# -------------------------------------------------------------')



