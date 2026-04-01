import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,EELS,Near_Field

Z = 25
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

omega_p = 8.96
gamma = 0.073
eps_inf = 9.7 #high frequency dielectric constant, units: eps_0
#eps_inf = 1 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(2.3,3.0,200) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega*8065.54/1e7

e = 1.602e-19
m = 9.109e-31
#V = 60e3
V = 100e3
c = 299792458
#E = e*V
#v = (2*e*V/m)**0.5 / c
#v = (1 - (m*c**2/(E + m*c**2)))**0.5
v = (1 - (1 + e*V/(m*c**2))**-2)**0.5
#v = 0.72


debug = []
#debug.append("ef_time")
debug.append("time")
#debug.append("guess")

#t0 = time.time()
#eels = EELS(box,eps_p,omega,v,eps_m,radius=R,ef_method="direct",cutoff=500,
            #split_frac=1/0.5,split_dist=1,debug=debug,retarded=True)
#eels, dips,poss, E = eels.compute(e_pos,points)
#print(time.time() - t0)

#omega = omega*1e7/8065.54

#peaks = []
#plt.figure(figsize=(8,4),layout="tight")
#for i in range(len(eels)):
    #max_idx = argrelmax(eels[i])[0]
    #peaks.append(max_idx)
    #print(max_idx)
    #print(omega[max_idx])
    #plt.plot(omega,eels[i]/np.max(eels[i]),"k-")


#p = dips[0]
#E = E[0]
#p = np.save("ps.npy",p)
#E = np.save("Es.npy",E)
#p = np.load("ps.npy")
#E = np.load("Es.npy")
#E = E/np.linalg.norm(E,axis=-1)[:,:,None]
#p = p[:,:,2]*E[:,:,2]
#plt.plot(omega,np.real(p))
#plt.show()
#plt.plot(omega,np.imag(p))
#plt.show()
#omega = omega/1e7*8065.54

p2 = np.copy(points)
p2[:,2] += 1*d
points = np.concatenate([points,p2],axis=0)

t0 = time.time()
print(omega)
eels = EELS(box,eps_p,omega,v,eps_m,radius=R,ef_method="direct",cutoff=500,
            split_frac=1/0.5,split_dist=1,debug=debug,retarded=True)
eels, dips,poss, E = eels.compute(e_pos,points)
print(time.time() - t0)
np.save("eels.npy",eels)
np.save("poss.npy",poss)
np.save("dips.npy",dips)
np.save("E.npy", E)

eels = np.load("eels.npy")
poss = np.load("poss.npy")
dips = np.load("dips.npy")
E = np.load("E.npy")

omega = omega*1e7/8065.54

peaks = []
for i in range(len(eels)):
    max_idx = argrelmax(eels[i])[0]
    peaks.append(max_idx)
    print(max_idx)
    print(omega[max_idx])
    plt.plot(omega,eels[i]/np.max(eels[i]),"k--")
    plt.ylabel("$\\Gamma_{EELS}$",fontsize=20)
    plt.xlabel("E (eV)",fontsize=20)
    plt.vlines(2.644,0,1,"b")
    plt.vlines(2.7,0,1,"g")
    plt.xlim(omega[0],omega[-1])
plt.show()


plt_func = lambda E: np.real(E)
for i in range(len(eels)):
    E = E[i]
    p = dips[i]
    E[np.abs(E)!=0] = E[np.abs(E)!=0]/np.abs(E)[np.abs(E)!=0]
    p = p*E
    #flag = points[:,2] == 0
    p = p[:,:,0]
    plt.plot(omega,np.real(p))
    plt.show()
    plt.plot(omega,np.imag(p))
    plt.show()


    fig,axs = plt.subplots(2,1,figsize=(5,7.5),gridspec_kw={"height_ratios":[1,0.5]})
    p = dips[i][peaks[i][0]]
    off = 1

    E0 = np.array([0,0,0])
    nf = Near_Field(box,E0,radius=R,omega=omega[peaks[i][0]]*8065.54/1e7,dip=p,cutoff=500,dip_pos=points,method="direct",retarded=True)
    x = 1.15*np.linspace(np.min(points[:,0]),np.max(points[:,0]),200)
    y = 1.15*np.linspace(np.min(points[:,1]),np.max(points[:,1]),200)
    xy = np.array(np.meshgrid(x,y)).T
    L0 = len(xy)
    xy = xy.reshape(-1,2)
    L1 = len(xy)
    fpts = np.concatenate([xy,off+np.zeros(L1)[:,None]],axis=-1)
    nf.set_field_points(fpts)
    E = nf.calculate(False)
    xy = xy.reshape(L0,-1,2)
    E = E.reshape(L0,-1,3)
    E = plt_func(E)
    y,x = xy.T

    ax = axs[0]
    ax.pcolormesh(x,y,E[:,:,2],cmap = plt.cm.bwr,vmin=-10,vmax=10)

    E0 = np.array([0,0,0])
    print(omega[peaks[i][0]])
    nf = Near_Field(box,E0,radius=R,omega=omega[peaks[i][0]]*8065.54/1e7,dip=p,cutoff=500,dip_pos=points,method="direct",retarded=True)
    x = 1.15*np.linspace(np.min(points[:,0]),np.max(points[:,0]),200)
    z = 1.5*np.linspace(np.min(points[:,2]),np.max(points[:,2]),200) - d/4
    x,z = np.meshgrid(x,z)
    L0 = len(x)
    x = x.reshape(-1,1)
    z = z.reshape(-1,1)
    L1 = len(x)
    fpts = np.concatenate([x,off+np.zeros(L1)[:,None],z],axis=-1)
    nf.set_field_points(fpts)
    E = nf.calculate(False)
    x = x.reshape(L0,-1)
    z = z.reshape(L0,-1)
    E = E.reshape(L0,-1,3)
    E = plt_func(E)

    ax = axs[1]
    ax.pcolormesh(x,z,E[:,:,1],cmap = plt.cm.bwr,vmin=-50,vmax=50)
    plt.show()
#
#
    #E0 = np.array([0,0,0])
    #nf = Near_Field(box,E0,radius=R,omega=omega[peaks[i][0]]*8065.54/1e7,dip=p,cutoff=500,dip_pos=points,method="direct",retarded=True)
    #fpts = np.copy(points)
    #sq = 0.2*np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])*d
    #fpts = sq[:,None,:] + fpts[None,:,:]
    #fpts = fpts.reshape(-1,3)
    #nf.set_field_points(fpts)
    #E = nf.calculate(False)
    #E = np.imag(E)
    #fpts = fpts.reshape(6,-1,3)
    #E = E.reshape(6,-1,3)
    
    #pnorm = np.linalg.norm(np.abs(p),axis=-1)
    #print(pnorm)
    #p = np.real(d*p/pnorm[:,None])
    #print(np.max(p))
    #ax = axs[0]
    #ax.set_yticks([])
    #ax.set_xticks([])
    #ax.quiver(*points[:,:2].T,*p[:,:2].T,pivot="middle")
    #ax = axs[1]
    #flag = [True,False,True]
    #pflag = points[:,1] == 0
    #pts = points[pflag]
    #p = p[pflag]
    #ax.quiver(*pts[:,flag].T,*p[:,flag].T,pivot="middle")
    #ax.set_yticks([])
    #ax.set_xticks([])
    #ax.set_ylim(-0.5*d,1.5*d)
    #plt.show()
    #for j in range(6):
        #ax = axs[0]
        #ax.scatter(*fpts[j,:,:2].T,c=E[j,:,2],cmap=plt.cm.bwr,vmin=-50,vmax=50)
        #ax.set_yticks([])
        #ax.set_xticks([])

        #ax = axs[1]
        #p = p[pflag]
        #F = fpts[j][pflag][:,flag]
        #e = E[j][pflag,2]
        #ax.scatter(*F.T,c=e,cmap=plt.cm.bwr,vmin=-50,vmax=50)
    #plt.show()

