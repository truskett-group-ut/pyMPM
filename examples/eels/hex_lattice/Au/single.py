import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,MPM,Near_Field

Z = 20
points= np.array([[0,0,0]])
box = np.array([1000,1000,1000])

d = 6.84
d_opt = 5
R = 0.5*d_opt

e_pos = np.array([[0.5,0]])*d
points = (points-points[0])*d
box = box*d

omega_p = 8.96
gamma = 0.073
eps_inf = 9.7 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(2.3,3.0,200) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega*8065.54/1e7


debug = []
#debug.append("ef_time")
#debug.append("time")
#debug.append("guess")

t0 = time.time()
mpm = MPM(box,eps_p,eps_m,omega=omega,radius=R,ef_method="direct",solver_method="bicgstab",cutoff=500,debug=debug,retarded=True,E0=[[1+0j,0,0]],quiet=True)
mpm.compute(points)
alpha = mpm.get_eff_polarizability()
p = mpm.get_dipoles()

print(alpha.shape)
print(omega.shape)
plt.plot(omega/8065.54*1e7,np.real(alpha[:,0]))
plt.plot(omega/8065.54*1e7,np.imag(alpha[:,0]))
plt.show()

idx = np.argmax(np.imag(alpha[:,0]))
p = np.array([p[idx,:]])

fig,axs = plt.subplots(2,1,figsize=(4,8))
plt_func = lambda E: np.real(E)

E0 = np.array([0,0,0])
nf = Near_Field(box,E0,radius=R,omega=omega[idx],dip=p,cutoff=500,
            dip_pos=points,method="direct",retarded=True)
x = np.linspace(-2,2,100)
y = np.linspace(-2,2,100)
xy = np.array(np.meshgrid(x,y)).T
L0 = len(xy)
xy = xy.reshape(-1,2)
L1 = len(xy)
fpts = np.concatenate([xy,0+np.zeros(L1)[:,None]],axis=-1)
nf.set_field_points(fpts)
E = nf.calculate(False)

xy = xy.reshape(L0,-1,2)
E = E.reshape(L0,-1,3)
E = plt_func(E)
y,x = xy.T

ax = axs[0]
ax.pcolormesh(x,y,E[:,:,2],cmap = plt.cm.bwr)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("x",fontsize=20)
ax.set_ylabel("y",fontsize=20)

E0 = np.array([0,0,0])
nf = Near_Field(box,E0,radius=R,omega=omega[idx],dip=p,cutoff=500,
            dip_pos=points,method="direct",retarded=True)

x = np.linspace(-2,2,100)
z = np.linspace(-2,2,100)
x,z = np.meshgrid(x,z)
L0 = len(x)
x = x.reshape(-1,1)
z = z.reshape(-1,1)
L1 = len(x)
fpts = np.concatenate([x,0+np.zeros(L1)[:,None],z],axis=-1)
nf.set_field_points(fpts)
E = nf.calculate(False)
x = x.reshape(L0,-1)
z = z.reshape(L0,-1)
E = E.reshape(L0,-1,3)
E = plt_func(E)

ax = axs[1]
ax.pcolormesh(x,z,E[:,:,1],cmap = plt.cm.bwr)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("x",fontsize=20)
ax.set_ylabel("z",fontsize=20)
plt.show()





