import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelmax
from pyMPM import drude_dielectric,EELS

#---- Set Lattice ----{{{
L = 100
N = 75
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

e_pos = np.array([0.5,0])*d
e_pos = (points[0,:2]+np.array([0.5,0]))*d
points = points*d
box = box*d
print(len(points))

omega_p = 11886
gamma = 845
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
eps_m = 1

omega = np.linspace(2000,9000,112) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0

omega = omega/1e7 #convert omega from cm^-1 to nm^-1

v = 0.45

#t0 = time.time()
#eels = EELS(box,eps_p,omega,v,eps_m,radius=R,method = "direct",cutoff=100)
#eels, dips, E = eels.compute(e_pos,points)
#print(time.time() - t0)
#np.save("hex_eels.npy",eels)
#np.save("hex_dips.npy",dips)
#np.save("hex_E.npy", E)

eels = np.load("hex_eels.npy")
dips = np.load("hex_dips.npy")
E = np.load("hex_E.npy")

omega = omega*1e7/8065.54


#plt.plot(omega,dips[...,0].imag)
#plt.show()
#plt.plot(omega,dips[...,1].imag)
#plt.show()
#plt.plot(omega,dips[...,2].imag)
#plt.show()

max_idx = np.argmax(eels)
plt.plot(omega,eels)
plt.ylabel("$\\Gamma_{EELS}$",fontsize=20)
plt.xlabel("E (eV)",fontsize=20)
plt.ylim(plt.ylim())
plt.vlines(omega[max_idx],0,100,color="k",linestyle="--")
plt.vlines(omega[max_idx+15],0,100,color="k",linestyle="--")
plt.show()

flag = np.all(c_points[:,:2]>-10,axis=1) & np.all(c_points[:,:2]<10,axis=1)
pts = c_points[flag]
#for i in range(len(omega)):
    #plt.figure(figsize=(5,5))
    #p = dips[i,flag,:]
    #p_norm = np.max(np.linalg.norm(p,axis=-1))
    #p = p/p_norm
    #Ei = E[i,flag].imag

    #scale = 0.9
    #plt.plot(0.5,0,"ro")
    #plt.quiver(*pts[:,:2].T,Ei[:,0],Ei[:,2],pivot="mid",color="k",scale=scale**-1,scale_units="x",
                    #units="xy",width=0.1,headwidth=3,headlength=3,headaxislength=3,
                    #minshaft=1.2)
    #plt.show()


for i in [max_idx,max_idx+15]:
    L = 5
    H = 7.5
    fig = plt.figure(figsize = (L,H))
    x = 0.5
    y = 0.5
    Lp = 4
    axb = fig.add_subplot((x/L,y/H,Lp/L,0.333*Lp/H))
    y = 2.25
    axt = fig.add_subplot((x/L,y/H,Lp/L,Lp/H))

    flag = np.all(c_points[:,:2]>-5,axis=1) & np.all(c_points[:,:2]<5,axis=1)
    pts = c_points[flag]
    p = dips[i,flag,:]
    p_norm = np.max(np.linalg.norm(np.abs(p),axis=-1))
    print(p_norm)
    p = np.real(p)/p_norm

    scale = 0.9
    axt.plot(0.5,0,"ro")
    axt.quiver(*pts[:,:2].T,*p[:,:2].T,pivot="mid",color="k",scale=scale**-1,scale_units="x",
                    units="xy",width=0.1,headwidth=3,headlength=3,headaxislength=3,
                    minshaft=1.2)
    axt.set_title(f"E = {omega[i]:.2}",fontsize=20)
    axt.set_xticks([])
    axt.set_yticks([])

    flag = ((c_points[:,0]>-5) & (c_points[:,0]<5)) & (c_points[:,1] == 0)
    pts = c_points[flag]
    p = dips[i,flag,:]
    p = np.real(p)/p_norm
    axb.plot(0.5,0,"ro")
    axb.quiver(pts[:,0],0,p[:,0],p[:,2],pivot="mid",color="k",
                    scale=scale**-1,scale_units="x",
                    units="xy",width=0.1,headwidth=3,headlength=3,headaxislength=3,
                    minshaft=1.2)
    axb.set_xticks([])
    axb.set_yticks([])


    plt.show()
