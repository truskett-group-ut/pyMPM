import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,EELS

L = 100
N = 75
#---- Set Lattice ----{{{
xstep = 1
Lx =  np.floor(L / xstep)
Lx += 1 if Lx%2 == 0 else 0
a1 = np.array([xstep,0,0])

ystep = 2*np.sin(np.pi/3)
Ly =  np.floor(L / ystep)
Ly += 1 if Ly%2 == 0 else 0
a2 = np.array([0,ystep,0])

b1 = np.array([np.cos(np.pi/3),np.sin(np.pi/3),0])

points = []
for i in np.arange(Lx+1):
    for j in np.arange(Ly+1):
        points.append(i*a1+j*a2)

box = np.array([(Lx+1)*xstep,(Ly+1)*ystep,L])
points = np.array(points)
points = np.concatenate([points,points+b1])

c_points = (points[:,:2] - box[:2]/2)
dist = np.linalg.norm(c_points,axis=1)
dist = np.round(dist,4)

flag = np.argsort(dist)
dist = dist[flag]
points = points[flag]
c_points = c_points[flag]

if dist[0] != 0:
    raise Exception("No middle point!")

coords = np.unique(dist)
coords = np.sort(coords)

flag = dist <= coords[N-1]
points = points[flag]
dist = dist[flag]
c_points = c_points[flag]
coords = coords[:N]
box = np.array([1000,1000,20])
#}}}
d = 25+2.6
d_opt = 25
R = 0.5*d_opt

#Z = 20
#box = np.array([1000,1000,Z])
#pts = np.loadtxt("hex_coords.txt")
#flag = pts[:,0]<=10
#pts = pts[flag] 
#points = np.hstack([pts[:,1:],np.zeros_like(pts[:,0])[:,None]])

e_pos = np.array([[0,0]])*d
points = (points-points[0])*d
box = box*d
print(len(points))

omega_p = 11886
gamma = 845
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(1000,10000,200) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0

omega = omega/1e7 #convert omega from cm^-1 to nm^-1

v = 0.45

debug = []
#debug.append("ef_time")
#debug.append("time")
#debug.append("guess")

t0 = time.time()
eels = EELS(box,eps_p,omega,v,eps_m,radius=R,method="direct",cutoff=100,
            split_frac=1/0.5,split_dist=1,debug=debug)
eels, dips,poss, E = eels.compute(e_pos,points)
print(time.time() - t0)
#np.save("eels.npy",eels)
#np.save("poss.npy",poss)
#np.save("dips.npy",dips)
#np.save("E.npy", E)

#eels = np.load("eels.npy")
#poss = np.load("poss.npy")
#dips = np.load("dips.npy")
#E = np.load("E.npy")

omega = omega*1e7/8065.54

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
        norm = np.max(np.linalg.norm(np.abs(dip[j]),axis=-1))
        px,py,pz = np.imag(dip[j].T)/norm

        #ax.quiver(x,y,z,px,py,pz,pivot="middle")
        ax.quiver(x,y,z,px,py,pz,normalize=True,length=0.2,pivot="middle")
        #norm = np.max(np.linalg.norm(np.abs(e[j]),axis=-1))
        #ex,ey,ez = np.real(e[j].T)/norm
        #ax.quiver(x,y,z,ex,ey,ez,pivot="middle")
        #ax.quiver(x,y,z,ex,ey,ez,normalize=True,length=0.2,pivot="middle")
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(-2.5,2.5)
        ax.set_zlim(-2.5,2.5)
        plt.show()

