import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyMPM import drude_dielectric,EELS

d = 25+2.6
d_opt = 25
R = 0.5*d_opt

box = np.array([20,20,20])*d
points = np.array([[-0.5,0,0],
                   [0.5,0,0]])*d

#ds = 0.25
#x = np.arange(-1.1,0+ds,ds)
#y = np.arange(-0.6,0+ds,ds)
#xy = np.array(np.meshgrid(x,y)).T * d
#e_pos = xy.reshape(-1,2)

e_pos = np.array([[0,0],[0.75,0],[1,0]])*d

omega_p = 11886
gamma = 845
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
eps_m = 1


omega = np.linspace(2000,9000,112) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega/1e7 #convert omega from cm^-1 to nm^-1


v = 0.45 #in units of c (speed of light)


t0 = time.time()
eels_direct = EELS(box,eps_p,omega,v,eps_m,radius=R,method="direct",cutoff=100)
eels_d, dips, E = eels_direct.compute(e_pos,points)
#eels_s = EELS(box,eps_p,omega,v,eps_m,radius=R,xi=0.5,method="ewald")
#eels_s, dips_s, E_s = eels_s.compute(e_pos,np.array([[0.5,0,0]]))


omega = omega*1e7/8065.54
#plt.plot(omega,eels_s[1]/np.max(eels_s[1]),label="Single NC")
for i,ep in enumerate(e_pos):
    plt.plot(omega,eels_d[i]/np.max(eels_d[i]),label=["Center","Quarter","Side"][i])
plt.ylabel("$\\Gamma_{EELS}$",fontsize=20)
plt.xlabel("E (eV)",fontsize=20)
plt.legend(fontsize=14,framealpha=0)
plt.show()

#for i,ep in enumerate(e_pos):
    #for j in range(len(omega)):
        #print(points)
        #plt.quiver(*points[:,2:],dips[j,:,2:])


