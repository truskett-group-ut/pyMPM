import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,EELS

box = np.array([1000,1000,20])
points = np.array([[0,0,0]])
e_pos = np.array([[0,0]])
d = 25
d_opt = 25
R = 0.5*d_opt

e_pos = e_pos*d
points = points*d
box = box*d

#eps_inf = 1.9
#eps_m = 2.13
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
eps_m = 1
omega = np.linspace(1000,10000,200) #Nw planewave wavenumbers, units: cm^-1
omega_p = 11886
gamma = 845
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0


v = 0.45
#debug = False
#debug = "ef_time"
debug = "time"

omega = omega/1e7 #convert omega from cm^-1 to nm^-1
t0 = time.time()
eels = EELS(box,eps_p,omega,v,eps_m,radius=R,method="direct",cutoff=100,debug=debug,split_frac=1/8)
eels, dips, poss, E = eels.compute(e_pos,points)
print(time.time() - t0)
omega = omega*1e7/8065.54 

np.save("single_eels.npy",eels)
np.save("single_dips.npy",dips)
np.save("single_poss.npy",poss)
np.save("single_E.npy", E)

eels = np.load("single_eels.npy")
dips = np.load("single_dips.npy")
poss = np.load("single_poss.npy")
E = np.load("single_E.npy")

#plt.plot(omega,dips[...,0].imag)
#plt.show()
#plt.plot(omega,dips[...,1].imag)
#plt.show()
#plt.plot(omega,dips[...,2].imag)
#plt.show()

peaks = []
for i in range(len(eels)):
    max_idx = argrelmax(eels[i])[0]
    peaks.append(max_idx)
    print(max_idx)
    print(omega[max_idx])
    plt.plot(omega,eels[i])
    plt.ylabel("$\\Gamma_{EELS}$",fontsize=20)
    plt.xlabel("E (eV)",fontsize=20)
plt.show()

for i in range(len(eels)):
    pos = poss[i]
    dip = dips[i]
    e = E[i]

    for j in peaks[i]:
        ax = plt.figure().add_subplot(projection="3d")
        x,y,z = pos.T
        px,py,pz = np.imag(dip[j].T)
        #ax.quiver(x,y,z,px,py,pz,pivot="middle")
        ax.quiver(x,y,z,px,py,pz,normalize=True,length=0.2,pivot="middle")
        #ex,ey,ez = np.imag(e[j].T)
        #ax.quiver(x,y,z,ex,ey,ez,pivot="middle")
        #ax.quiver(x,y,z,ex,ey,ez,normalize=True,length=0.2,pivot="middle")
        plt.show()

