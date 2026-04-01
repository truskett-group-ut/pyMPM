import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,MPM

Z = 20
box = np.array([1000,1000,Z])
pts = np.loadtxt("hex_coords.txt")
flag = pts[:,0]<=4
pts = pts[flag]
points = np.hstack([pts[:,1:],np.zeros_like(pts[:,0])[:,None]])
d = 6.84
d_opt = 5
R = 0.5*d_opt

e_pos = np.array([[0.5,0]])*d
points = (points-points[0])*d
box = box*d
print(len(points))

omega_p = 8.96
gamma = 0.073
eps_inf = 9.7 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(2.3,3.0,200) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega*8065.54/1e7


debug = []
#debug.append("ef_time")
debug.append("time")
#debug.append("guess")

t0 = time.time()
mpm = MPM(box,eps_p,eps_m,omega=omega,radius=R,ef_method="direct",solver_method="bicgstab",cutoff=500,debug=debug,retarded=True,E0=[[1,0,0]])
mpm.compute(points)
alpha = mpm.get_eff_polarizability()
p = mpm.get_dipoles()
print(time.time() - t0)
np.save("alpha.npy",alpha)
np.save("p.npy",p)

alpha = np.load("alpha.npy")
p = np.load("p.npy")

omega = omega*1e7/8065.54

plt.figure(figsize=(8,4),layout="tight ")
plt.plot(omega,np.imag(alpha[:,0])/np.max(np.imag(alpha[:,0])),'k-')
plt.vlines(2.56,0,1,"r")

p2 = np.copy(points)
p2[:,2] += 1*d
points = np.concatenate([points,p2],axis=0)

omega = omega*8065.54/1e7

t0 = time.time()
mpm = MPM(box,eps_p,eps_m,omega=omega,radius=R,ef_method="direct",solver_method="gmres",cutoff=500,debug=debug,retarded=True,E0=[[1,0,0]])
mpm.compute(points)
alpha = mpm.get_eff_polarizability()
p = mpm.get_dipoles()
print(time.time() - t0)
np.save("alpha.npy",alpha)
np.save("p.npy",p)

alpha = np.load("alpha.npy")
p = np.load("p.npy")

omega = omega*1e7/8065.54

plt.plot(omega,np.imag(alpha[:,0])/np.max(np.imag(alpha[:,0])),'k--')
plt.ylabel("$\\alpha$",fontsize=20)
plt.xlabel("E (eV)",fontsize=20)
plt.show()


idx = np.argmax(np.imag(alpha[:,0]))

p = np.imag(p[idx,:])
p = p / np.max(np.linalg.norm(p,axis=1))

fig,axs = plt.subplots(2,1,figsize=(5,7.5),gridspec_kw={"height_ratios":[1,0.5]})
ax = axs[0]
ax.quiver(*points[:,:2].T,*p[:,:2].T,pivot="middle")
ax.set_yticks([])
ax.set_xticks([])
ax = axs[1]
flag = [True,False,True]
pflag = points[:,1] == 0
pts = points[pflag]
p = p[pflag]
ax.quiver(*pts[:,flag].T,*p[:,flag].T,pivot="middle")
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylim(-0.5*d,1.5*d)
plt.show()

