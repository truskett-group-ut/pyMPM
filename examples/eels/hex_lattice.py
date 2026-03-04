import numpy as np
import matplotlib.pyplot as plt
import time
from pyMPM import drude_dielectric,EELS

L = 40
N = 24

#---- Set Lattice ----{{{
#xstep = 1
#Lx =  np.floor(L / xstep)
#Lx += 1 if Lx%2 == 0 else 0
#a1 = np.array([xstep,0,0])

#ystep = 2*np.sin(np.pi/3)
#Ly =  np.floor(L / ystep)
#Ly += 1 if Ly%2 == 0 else 0
#a2 = np.array([0,ystep,0])
#
#b1 = np.array([np.cos(np.pi/3),np.sin(np.pi/3),0])
#
#points = []
#for i in np.arange(Lx+1):
    #for j in np.arange(Ly+1):
        #points.append(i*a1+j*a2)
#
#box = np.array([(Lx+1)*xstep,(Ly+1)*ystep,L/2])
#points = np.array(points)
#points = np.concatenate([points,points+b1])
#
#c_points = (points[:,:2] - box[:2]/2)
#dist = np.linalg.norm(c_points,axis=1)
#dist = np.round(dist,4)
#
#flag = np.argsort(dist)
#dist = dist[flag]
#points = points[flag]
#c_points = c_points[flag]
#
#if dist[0] != 0:
    #raise Exception("No middle point!")
#
#coords = np.unique(dist)
#coords = np.sort(coords)
#
#flag = dist <= coords[N-1]
#points = points[flag]
#dist = dist[flag]
#coords = coords[:N]
#}}}
box = np.array([40,40,40])
pts = np.loadtxt("hex_coords.txt")
flag = pts[:,0]<=10
pts = pts[flag]
points = np.hstack([pts[:,1:],np.zeros_like(pts[:,0])[:,None]])
#e_pos = np.array([0.5,0])
#e_pos = np.array([0.44,0])

#box = np.array([10,10,20])
#points = np.array([[0,0,0]])
#points = np.array([[0,0,0],
                   #[1,0,0]])
#e_pos = np.array([11,0])


#e_pos = box[:2] / 2
#e_pos[0] = e_pos[0] + 0.5

#plt.figure(figsize=(5,5))
#plt.plot(*points[:,:2].T,'bo',markersize=4)
#plt.plot(*e_pos,"ro",markersize=4)
#plt.hlines(box[1],0,box[0],'k')
#plt.hlines(0,0,box[0],'k')
#plt.vlines(box[0],0,box[1],'k')
#plt.vlines(0,0,box[1],'k')
#plt.show()


#omega_p = 12313 #plasma frequency, units: cm^-1
omega_p = 11886
#gamma = 681 #damping coefficient, units: cm^-1
gamma = 845
#eps_inf = 4 #high frequency dielectric constant, units: eps_0
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
omega = np.linspace(4000,9000,80) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0

omega = omega/1e7 #convert omega from cm^-1 to nm^-1

d = 25+2.6
d_opt = 25
#d = 11.7
#d_opt = 10
#eps_m = 2.13
eps_m = 1
v = 0.45

#points[:,2] = 0

t0 = time.time()
eels = EELS(box*d,eps_p,omega,v,eps_m,radius=0.5*d_opt)
eels, dips, E = eels.compute(e_pos*d,points*d)
print(time.time() - t0)
np.save("hex_eels.npy",eels)
np.save("hex_dips.npy",dips)
np.save("hex_E.npy", E)

eels = np.load("hex_eels.npy")
dips = np.load("hex_dips.npy")
E = np.load("hex_E.npy")
dip_norms = np.linalg.norm(np.real(dips),axis=-1)
dips = dips/np.max(dip_norms)
omega = omega*1e7
c = 2.99e10
h = 6.625e-35
toEV = 6.242e18

plt.plot(omega/8065.54,eels)
plt.show()



plt.plot(omega,dips[:,:,0])#/(E/np.abs(E))[...,0]).imag)
plt.show()
plt.plot(omega,dips[:,:,1])#/(E/np.abs(E))[...,1]).imag)
plt.show()
plt.plot(omega,dips[:,:,2])#/(E/np.abs(E))[...,2]).imag)
plt.show()


#for i in range(len(omega)):
    #plt.figure(figsize=(5,5))
    #plt.plot(*e_pos,"ro")
    #plt.quiver(*points[:,:2].T,dips[i,:,0].T,dips[i,:,1].T)
    #plt.quiver(*points[:,:2].T,0,np.real(dips[i,:,2].T),pivot="mid")
    #plt.xlim(0,L)
    #plt.ylim(0,L)
    #plt.show()
