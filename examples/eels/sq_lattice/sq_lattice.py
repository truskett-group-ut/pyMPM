import numpy as np
import matplotlib.pyplot as plt
import time
from pyMPM import drude_dielectric,EELS

d = 11.7
N = 20
L = d*N
A = np.arange(-L/2,L/2,d) + d/2
ex = 25
y,x = np.meshgrid(A,A)
pos = np.array([x,y,np.zeros_like(x)]).T
pos = pos.reshape(N**2,3) #units: nm
box = np.array([ex*d+L, ex*d+L, ex*d])
pos[:,:2] += box[:2]/2
e_pos = box[:2]/2

omega_p = 12313 #plasma frequency, units: cm^-1
gamma = 681 #damping coefficient, units: cm^-1
eps_inf = 4 #high frequency dielectric constant, units: eps_0
omega = np.linspace(2000,6000,80) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0

omega = omega/1e7 #convert omega from cm^-1 to nm^-1

d_opt = 10
eps_m = 2.13
v = 0.45

#pos = np.array([[(N/2+0.5)*d, (N/2+0.5)*d,0]])

#t0 = time.time()
#mpm = EELS(box,eps_p,omega,v,eps_m,radius=d_opt/2)
#eels, dips, E = mpm.compute(e_pos,pos)
#print(time.time() - t0)
#np.save("eels.npy",eels)
#np.save("dips.npy",dips)
#np.save("E.npy", E)

eels = np.load("eels.npy")
dips = np.load("dips.npy")
E = np.load("E.npy")
#dip_norms = np.linalg.norm(np.real(dips),axis=-1)
#dips = dips/np.max(dip_norms)
omega = omega*1e7


plt.plot(omega,eels)
plt.show()



plt.plot(omega,(dips[:,:,0]/(E/np.abs(E))[...,0]).imag)
plt.show()
plt.plot(omega,(dips[:,:,1]/(E/np.abs(E))[...,1]).imag)
plt.show()
plt.plot(omega,(dips[:,:,2]/(E/np.abs(E))[...,2]).imag)
plt.show()


#plt.figure(figsize=(5,5))
#for i in range(len(omega)):
    #plt.plot(*e_pos,"ro")
    #plt.quiver(*pos[:,:2].T,dips[i,:,0].T,dips[i,:,1].T)
    #plt.quiver(*pos[:,:2].T,0,np.real(dips[i,:,2].T),pivot="mid")
    #plt.xlim(0,L)
    #plt.ylim(0,L)
    #plt.show()
