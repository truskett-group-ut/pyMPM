import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,EELS,Near_Field

Z = 25
box = np.array([1000,1000,Z])
pts = np.loadtxt("hex_coords.txt")
flag = pts[:,0]<=4
pts = pts[flag]
points = np.hstack([pts[:,1:],np.zeros_like(pts[:,0])[:,None]])
print(points[37])

d = 6.84
d_opt = 5
R = 0.5*d_opt

e_pos = np.array([[0.5,0]])*d
points = (points-points[0])*d
box = box*d

omega_p = 8.96
gamma = 0.073
eps_inf = 9.7 #high frequency dielectric constant, units: eps_0
#eps_inf = 1 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(2.3,3.0,200) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega*8065.54/1e7
print(omega[0]**-1,omega[-1]**-1)

e = 1.602e-19
m = 9.109e-31
#V = 60e3
V = 100e3
c = 299792458
v = (1 - (1 + e*V/(m*c**2))**-2)**0.5


debug = []
#debug.append("ef_time")
#debug.append("time")
#debug.append("guess")
quiet = True

#t0 = time.time()
#eels = EELS(box,eps_p,omega,v,eps_m,radius=R,ef_method="direct",cutoff=500,
            #split_frac=1/0.5,split_dist=1,debug=debug,retarded=True,quiet=quiet)
#eels, dips,poss, E = eels.compute(e_pos,points)
#print(time.time() - t0)

#p = dips[0]
#E = E[0]
#p = np.save("ps.npy",p)
#E = np.save("Es.npy",E)
#eels = np.save("eels_s.npy",eels)
#poss_s = np.save("poss_s.npy",poss)
p = np.load("ps.npy")
E = np.load("Es.npy")
eels = np.load("eels_s.npy")
poss = np.load("poss_s.npy")
omega = omega*1e7/8065.54

peaks = []
for i in range(len(eels)):
    max_idx = argrelmax(eels[i])[0]
    peaks.append(max_idx)
    #plt.plot(omega,eels[i]/np.max(eels[i]),"k--")
    plt.plot(omega,eels[i],"k--")
    plt.ylabel("$\\Gamma_{EELS}$",fontsize=20)
    plt.xlabel("E (eV)",fontsize=20)
    #plt.vlines(2.644,0,1,"b")
    #plt.vlines(2.7,0,1,"g")
    plt.xlim(omega[0],omega[-1])
plt.show()

idx = peaks[0][0]
#for idx in range(len(omega)):
    #if omega[idx] > np.inf:
        #break

dips = [p]
flags = points[:,1] == 0
pts = points[flags][:,[0,2]]
for i in range(len(omega)):
    p = dips[0][i]
    p = p[flags][:,[0,2]].imag
    plt.quiver(*pts.T,*p.T,pivot = "middle")
    plt.show()

#Plot Field Dipoles
#dips = [p]
#plt_func = lambda E: np.imag(E)
#for i in range(len(eels)):
#
    #fig,axs = plt.subplots(2,1,figsize=(5,7.5),gridspec_kw={"height_ratios":[1,0.5]})
    #p = dips[i][idx]
    #off = 1
#
    #E0 = np.array([0,0,0])
    #nf = Near_Field(box,E0,radius=R,omega=omega[idx]*8065.54/1e7,dip=p,cutoff=500,dip_pos=points,method="direct",retarded=True)
    #x = 1.15*np.linspace(np.min(points[:,0]),np.max(points[:,0]),200)
    #y = 1.15*np.linspace(np.min(points[:,1]),np.max(points[:,1]),200)
    #xy = np.array(np.meshgrid(x,y)).T
    #L0 = len(xy)
    #xy = xy.reshape(-1,2)
    #L1 = len(xy)
    #fpts = np.concatenate([xy,off+np.zeros(L1)[:,None]],axis=-1)
    #nf.set_field_points(fpts)
    #E = nf.calculate(False)
    #xy = xy.reshape(L0,-1,2)
    #E = E.reshape(L0,-1,3)
    #E = plt_func(E)
    #y,x = xy.T
#
    #ax = axs[0]
    #Eplot = E[:,:,2]
    #norm = colors.TwoSlopeNorm(vcenter=0)
    #ax.pcolormesh(x,y,Eplot,cmap = plt.cm.bwr,norm=norm)
#
    #E0 = np.array([0,0,0])
    #nf = Near_Field(box,E0,radius=R,omega=omega[idx]*8065.54/1e7,dip=p,cutoff=500,dip_pos=points,method="direct",retarded=True)
    #x = np.linspace(-35,35,200)
    #z = np.linspace(-15,15,200)
    #x,z = np.meshgrid(x,z)
    #L0 = len(x)
    #x = x.reshape(-1,1)
    #z = z.reshape(-1,1)
    #L1 = len(x)
    #fpts = np.concatenate([x,off+np.zeros(L1)[:,None],z],axis=-1)
    #nf.set_field_points(fpts)
    #E = nf.calculate(False)
    #x = x.reshape(L0,-1)
    #z = z.reshape(L0,-1)
    #E = E.reshape(L0,-1,3)
    #E = plt_func(E)
#
    #ax = axs[1]
    #Eplot = E[:,:,1]
    #norm = colors.TwoSlopeNorm(vcenter=0)
    #ax.pcolormesh(x,z,Eplot,cmap = plt.cm.bwr,norm=norm)
    #plt.show()
