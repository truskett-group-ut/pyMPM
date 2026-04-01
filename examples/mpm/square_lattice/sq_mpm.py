import time
import pyMPM
import numpy as np
import matplotlib.pyplot as plt

d = 11.7 #particle diameter, units: nm

#Create particle positions and box
N = 11 
L = d*N
A = np.arange(0,L,d)
y,x = np.meshgrid(A,A)
pos = np.array([x,y,np.zeros_like(x)]).T
pos = pos.reshape(N**2,3) #units: nm
box = np.array([L,L,50*d]) #units: nm
np.save("pos.npy",pos)
np.save("box.npy",box)

#Test particle positions
#pos += d/2
#plt.subplots(figsize=(4,4))
#plt.hlines(0,0,L,color="k",linewidth=3)
#plt.hlines(L,0,L,color="k",linewidth=3)
#plt.vlines(0,0,L,color="k",linewidth=3)
#plt.vlines(L,0,L,color="k",linewidth=3)

#plt.plot(*pos[:,:2].T,"ro",markersize=3)
#plt.xticks([])
#plt.yticks([])
#plt.tight_layout()
#plt.show()

#Setup Particle Dielectric
omega_p = 12313 #plasma frequency, units: cm^-1
gamma = 681 #damping coefficient, units: cm^-1
eps_inf = 4 #high frequency dielectric constant, units: eps_0
omega = np.linspace(1000,7000,80) #Nw planewave wavenumbers, units: cm^-1
eps_p = pyMPM.drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0

# Run MPM
d_opt = 10 # optical core diameter, units: nm
eps_m = 2.13 # Medium dielectric constant, units: eps_0

t0 = time.time()
mpm = pyMPM.MPM(box,eps_p,radius=d_opt/2,eps_m=eps_m)
mpm.compute(pos)
print("MPM ran in:",time.time()-t0)

p = mpm.get_dipoles()
alpha_eff = mpm.get_eff_polarizability()
np.save("dipoles.npy",p)
np.save("polarizability.npy",alpha_eff)

plt.plot(omega,np.imag(alpha_eff[:,0,0]),label="In-plane")
plt.plot(omega,np.imag(alpha_eff[:,2,2]),label="Out-of-plane")
plt.xlim(np.flip(plt.xlim()))
plt.xlabel("$\\omega$ (cm$^{-1}$)",fontsize=18)
plt.ylabel("Im($\\alpha_{\\text{eff}}$)",fontsize=18)
plt.legend(fontsize=14,framealpha=0)
plt.tight_layout()
plt.show()

plt.plot(omega,np.real(alpha_eff[:,0,0]),label="In-plane")
plt.plot(omega,np.real(alpha_eff[:,2,2]),label="Out-of-plane")
plt.xlim(np.flip(plt.xlim()))
plt.xlabel("$\\omega$ (cm$^{-1}$)",fontsize=18)
plt.ylabel("Re($\\alpha_{\\text{eff}}$)",fontsize=18)
plt.legend(fontsize=14,framealpha=0)
plt.tight_layout()
plt.show()
