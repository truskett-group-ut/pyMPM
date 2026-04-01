import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyMPM import drude_dielectric,EELS

d = 25+2.6
d_opt = 25
R = 0.5*d_opt

box = np.array([10,10,10])*d
points = np.array([[-0.5,0,0],
                   [0.5,0,0]])*d

ds = 0.25
x = np.arange(-1.1,0+ds,ds)
y = np.arange(-0.6,0+ds,ds)
xy = np.array(np.meshgrid(x,y)).T * d
e_pos = xy.reshape(-1,2)

#e_pos = np.array([[0,0],[1,0]])*d

omega_p = 11886
gamma = 845
eps_inf = 2.25 #high frequency dielectric constant, units: eps_0
eps_m = 1


omega = np.linspace(2000,9000,112) #Nw planewave wavenumbers, units: cm^-1
eps_p = drude_dielectric(omega,gamma,omega_p,eps_inf) #particle drude dielectric as a function of wavenumber, units: eps_0
omega = omega/1e7 #convert omega from cm^-1 to nm^-1


v = 0.45 #in units of c (speed of light)


t0 = time.time()
eels = EELS(box,eps_p,omega,v,eps_m,radius=R,xi=0.5,method="direct")
eels, dips, E = eels.compute(e_pos,points)
print(time.time() - t0)
np.save("hex_eels.npy",eels)
np.save("hex_dips.npy",dips)
np.save("hex_E.npy", E)

eels = np.load("hex_eels.npy")
dips = np.load("hex_dips.npy")
E = np.load("hex_E.npy")

omega = omega*1e7
for i,ep in enumerate(e_pos):
    plt.plot(omega/8065.54,eels[i])
    plt.show()


#dip_comp = np.zeros_like(dips)
#dip_comp[...,2] = dips[...,2]
#t0 = time.time()
#eels = EELS(box,eps_p,omega,v,eps_m,radius=R)
#eels, dips, E = eels.compute(e_pos,points,dipoles=dip_comp)
#print(time.time() - t0)

#np.save("dimer_eels_Z.npy",eels)
#eels = np.load("dimer_eels_Z.npy")

#---- Play hyperspectra ----
#omega = omega*1e7
#eels = eels.reshape(*xy.shape[:-1],-1)
#Enorm = np.max(np.linalg.norm(np.abs(E),axis=-1),axis=1)
##eels = eels/Enorm[None,None,:]/(4*np.pi)
#idx = np.argwhere(eels==np.max(eels))[0]
#eels[*idx[:2],:] = np.nan
#idx = np.argwhere(eels==np.nanmin(eels))[0]
#eels[*idx[:2],:] = np.nan
#
#fig,ax = plt.subplots(figsize=(4*1.1/0.6*1.1,4))
#vmin = np.nanmin(eels)
#vmax = np.nanmax(eels)
##vmin = 0
##vmax = 10
#cmap = ax.pcolormesh(xy[...,0],xy[...,1],eels[...,0],vmin=vmin,vmax=vmax)
#cbar = fig.colorbar(cmap,ax=ax)
#cbar.set_label("$\\Gamma_{EELS}$",fontsize=14)
#
#def update(frame):
    #i = frame
    #ax.cla()
    #ax.pcolormesh(xy[...,0],xy[...,1],eels[...,i],vmin=vmin,vmax=vmax)
    #ax.scatter(*points[:,:2].T,color="r")
    #ax.set_title(f"Energy = {omega[frame]/8065.54:.2} eV")
#
#def onclick(event):
    #if onclick.paused:
        #onclick.paused = False
        #ani.resume()
    #else:
        #onclick.paused = True
        #ani.pause()
    #return
#
#onclick.paused = False
#fig.canvas.mpl_connect('button_press_event',onclick)
#ani = animation.FuncAnimation(fig,update,frames=len(omega))
#writer = animation.FFMpegWriter(fps=15)
#ani.save("animation.mp4",writer=writer)
#plt.show()

#plt.plot(omega,dips[:,:,0])#/(E/np.abs(E))[...,0]).imag)
#plt.show()
#plt.plot(omega,dips[:,:,1])#/(E/np.abs(E))[...,1]).imag)
#plt.show()
#plt.plot(omega,dips[:,:,2])#/(E/np.abs(E))[...,2]).imag)
#plt.show()


    #for i in range(len(omega)):
    #plt.figure(figsize=(5,5))
    #plt.plot(*ep,"ro")
    #plt.quiver(*points[:,:2].T,dips[0,:,0].T,dips[0,:,1].T)
    #plt.quiver(*points[:,:2].T,0,np.real(dips[i,:,2].T),pivot="mid")
    #plt.xlim(0,L)
    #plt.ylim(0,L)
    #plt.show()
