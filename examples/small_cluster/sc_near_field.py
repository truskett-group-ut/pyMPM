import numpy as np
from pyMPM import Near_Field

def sc_near_field():
    k = np.arange(0.01,0.81,0.01)
    box = np.array([30,30,30])
    positions = np.array(
                [[-1,2.8,0],
                [-4.6,2.3,0],
                [-3.05,1,0],
    
                [-1.35,-0.45,0],
                [0.4,-1.4,0],
    
                [4.3,1.2,0],
                [2.65,0.1,0],
                [4.5,-1,0],
    
                [-4.2,-1.05,0],
                [-2.75,-2.5,0]])

    C = np.load("cap.npy")
    p = np.load("dip.npy")

    dim = 1
    arg = np.argmax(k*np.imag(C[:,dim,dim]))
    p = p[arg,:,dim]

    E0 = np.array([0,1,0])
    e = Near_Field(box,E0,dip=p,dip_pos=positions,field_min=-5.6,field_max=5.6,num_field=500)
    pos,E = e.calculate()
    
    return pos,E,positions,p

if __name__ == "__main__":
    import time
    import matplotlib as mpl
    import matplotlib.pyplot as plt 
    import matplotlib.patches as patch

    start = time.time()

    pos,E,positions,dip = sc_near_field()

    dt = time.time() - start
    print(f"stratified_media ran in {dt} seconds")

    fig,ax = plt.subplots(1,1,figsize=(5.5,5))
    cmap = ax.pcolormesh(*pos.T,np.log(E),edgecolor="face",vmin=-1,cmap=plt.cm.inferno)
    plt.colorbar(cmap,ax=ax)
    for p in positions:
        ps = patch.Circle(p,0.99,facecolor="k")
        ax.add_patch(ps)
    
    p = np.imag(dip[:,:2])
    p0 = np.linalg.norm(p,axis=-1)
    pmax = np.max(p0)
    pmin = np.min(p0)
    p = p/pmax

    ax.quiver(positions[:,0],positions[:,1],
          p[:,0],p[:,1],
          color="w",pivot="mid",
          scale=0.5,scale_units="x",
          units="x",width=0.1,
          headlength=3,headwidth=3,headaxislength=3)

    ax.set_ylim(-5.6,5.6)
    ax.set_xlim(-5.6,5.6)

    plt.show()
