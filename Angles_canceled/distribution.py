import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
n = 2000


t0 = np.random.uniform(0,1)*np.pi
p0 = np.random.uniform(0,2)*np.pi


'''
xi1 = np.random.uniform(0, 1, size=n )
xi2 = np.random.uniform(0, 1, size=n )

theta = 2*np.arcsin(np.sqrt(xi1))
phi = 2*np.pi*xi2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta), zdir='z')
ax.plot([0,np.sin(t0)*np.cos(p0)], [0,np.sin(t0)*np.sin(p0)], [0,np.cos(t0)], zdir='z', color='r')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
ax.set_zticks([-1,0,1])

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.3, 1]))
'''

print('# -------------------------------------------------------------')

'''
dphi = 0.1
dtheta = 0.1



xi1 = np.random.uniform(-1, 1, size=n )
xi2 = np.random.uniform(-1, 1, size=n )
theta = np.arccos( np.cos(t0) - xi1*np.sin(t0)*np.sin(dtheta) )
#dphi = np.sqrt(0.1**2-(theta-t0)**2)
phi = dphi*xi2 + p0

to_change = [i for i,x in enumerate(theta) if x >= np.pi]
theta[to_change] -= np.pi
phi[to_change] += np.pi

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta), zdir='z')
ax2.plot([0,np.sin(t0)*np.cos(p0)], [0,np.sin(t0)*np.sin(p0)], [0,np.cos(t0)], zdir='z', color='r')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_zlabel(r'$z$')
ax2.set_xlim([-1,1])
ax2.set_ylim([-1,1])
ax2.set_zlim([-1,1])
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
ax2.set_zticks([-1,0,1])


#ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.3, 1]))


#print( abs(theta-t0) > dtheta, '\n', abs(phi-p0) > dphi)
#print( (theta-t0).T, (phi-p0).T )
print( np.sum(theta < 0), np.sum(theta > np.pi), np.sum(phi < 0), np.sum(phi > 2*np.pi))
'''

'''
t0 = np.random.uniform(0,1)*np.pi
p0 = np.random.uniform(0,2)*np.pi

phi = np.random.normal(p0, 0.1, size = n)
theta = np.random.normal(t0, 0.1, size = n)


fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta), zdir='z')
ax2.plot([0,np.sin(t0)*np.cos(p0)], [0,np.sin(t0)*np.sin(p0)], [0,np.cos(t0)], zdir='z', color='r')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
ax2.set_zlabel(r'$z$')
ax2.set_xlim([-1,1])
ax2.set_ylim([-1,1])
ax2.set_zlim([-1,1])
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
ax2.set_zticks([-1,0,1])


#ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.3, 1]))


#print( abs(theta-t0) > dtheta, '\n', abs(phi-p0) > dphi)
print( (theta-t0).T, (phi-p0).T )
'''

sigma = 0.3

points = np.zeros( (n+1,3) )
points[0] = np.array([np.sin(t0)*np.cos(p0), np.sin(t0)*np.sin(p0), np.cos(t0)])

print(f'Initial position ({t0}, {p0}) = ({points[0]})')
for k in range(n):
	p_new = np.random.normal(points[k],sigma, size=3)
	r = np.linalg.norm(p_new)

	points[k+1] = p_new/r


phi = np.arctan2(points[:,1],points[:,0]) 
phi[phi< 0 ] += 2*np.pi
theta = np.arctan2(np.sqrt(points[:,0]**2 + points[:,1]**2),points[:,2])


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta), zdir='z', marker='o', linewidth=0.5, markersize=3.0, linestyle='dotted')
ax1.plot([0,np.sin(t0)*np.cos(p0)], [0,np.sin(t0)*np.sin(p0)], [0,np.cos(t0)], zdir='z', color='r')
#ax1.scatter(np.sin(t0)*np.cos(p0), np.sin(t0)*np.sin(p0), np.cos(t0), zdir='z', color='r')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
ax1.set_zlim([-1,1])
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
ax1.set_zticks([-1,0,1])
ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 1.3, 1]))


'''
print(f'theta \t phi')
for k in range(n+1):
	print(f'{theta[k]:.3f} \t {phi[k]:.3f}')
'''
print('Angles outside range : ', theta[theta>np.pi] ,theta[theta<0],phi[phi<0], phi[phi > 2*np.pi])



plt.figure()
plt.plot(theta/np.pi, label = r'$\theta$')
plt.plot(phi/2/np.pi, label=r'$\phi$')
plt.xlabel('Iteration')
plt.ylabel(r'$\theta/\pi, \phi/2\pi$')
plt.legend()



plt.show();
