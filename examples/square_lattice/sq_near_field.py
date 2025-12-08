import time
import pyMPM
import numpy as np

d = 11.7 #particle diameter, units: nm
d_opt = 10 # optical core diameter, units: nm

#load positions and box
pos = np.load("pos.npy",)
box = np.load("box.npy",)

#Choose omega
dim = 2 #Polarization direction
alpha_eff = np.load("polarizability.npy")
omega_idx = np.argmax(np.imag(alpha_eff[:,dim,dim]))

#Load dipoles and set incident field
p = np.load("dipoles.npy")
p = p[omega_idx,:,dim]
E0 = np.zeros(3)
E0[dim] = 1

#Create field points
n_pts = 100
A = np.linspace(0,d,n_pts)
y,x = np.meshgrid(A,A)
field_points = np.array([x,y,np.zeros_like(x)]).T
field_points = field_points.reshape(n_pts**2,3)

#Calculate near field
t0 = time.time()
nf = pyMPM.Near_Field(box,E0,radius=d_opt/2)
nf.set_field_points(field_points)
nf.set_dipole_positions(pos)
nf.set_dipoles(p)
E = nf.calculate()
E = E.reshape(n_pts,n_pts)
print("Near field calculated in",time.time()-t0)

#Plot
import matplotlib.pyplot as plt
from matplotlib import patches
fig,ax = plt.subplots(figsize=(5,4))
plt.pcolormesh(x,y,E)
plt.yticks([])
plt.xticks([])
cb = plt.colorbar()
cb.set_label("$|E/E_0|$",fontsize=14)
for x in [0,d]:
    for y in [0,d]:
        c = patches.Circle((x,y),d_opt/2,color=plt.cm.Grays(0.5))
        ax.add_patch(c)
plt.tight_layout()
plt.show()
