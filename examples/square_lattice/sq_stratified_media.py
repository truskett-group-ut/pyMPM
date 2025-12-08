import time
import pyMPM
import numpy as np

d = 11.7 #particle diameter, units: nm
d_opt = 10 # optical core diameter, units: nm
pos = np.load("pos.npy",)
N = pos.shape[0]
box = np.load("box.npy",)
omega = np.linspace(1000,7000,80) #cm^-1
omega = omega / 1e7 #nm^-1

h = d
eps_m = 2.13
etaV = np.pi/6*d_opt**3*N / (box[0]*box[1]*h)

alpha_eff = np.load("polarizability.npy")
I = np.identity(3,dtype=bool)
eps_eff = eps_m*(1 + 3/4/np.pi*etaV*alpha_eff[:,I])

theta_max = 60*np.pi/180
num_thetas = 5
thetas = np.linspace(0,theta_max,num_thetas)

hs = [h,500e3]
eps = [eps_eff,11.7]
SM = pyMPM.Stratified_Media(omega,hs,eps=eps,eps_inf=1)

Rp,Tp = SM.compute_p(thetas,avg_interference=True)

import matplotlib.pyplot as plt
omega = omega*1e7

for i in range(num_thetas):
    plt.plot(omega,Rp[i],color=plt.cm.Blues(i/(num_thetas-1)*0.8 + 0.2),label=f"{int(np.round(180/np.pi*thetas[i]))}$^o$")
plt.ylabel("Reflectance",fontsize=18)
plt.xlabel("$\\omega$ (cm$^{-1}$)",fontsize=18)
plt.legend(title="Angle",fontsize=14,title_fontsize=16,framealpha=0)
plt.tight_layout()
plt.show()

for i in range(num_thetas):
    plt.plot(omega,Tp[i],color=plt.cm.Greens(i/(num_thetas-1)*0.8 + 0.2),label=f"{int(np.round(180/np.pi*thetas[i]))}$^o$")
plt.ylabel("Transmittance",fontsize=18)
plt.xlabel("$\\omega$ (cm$^{-1}$)",fontsize=18)
plt.legend(title="Angle",fontsize=14,title_fontsize=16,framealpha=0)
plt.tight_layout()
plt.show()

for i in range(num_thetas):
    plt.plot(omega,1-Rp[i]-Tp[i],color=plt.cm.Reds(i/(num_thetas-1)*0.8 + 0.2),label=f"{int(np.round(180/np.pi*thetas[i]))}$^o$")
plt.ylabel("Absorbance",fontsize=18)
plt.xlabel("$\\omega$ (cm$^{-1}$)",fontsize=18)
plt.legend(title="Angle",fontsize=14,title_fontsize=16,framealpha=0)
plt.tight_layout()
plt.show()
