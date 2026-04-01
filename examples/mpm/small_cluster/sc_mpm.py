import numpy as np
from pyMPM import MPM
from pyMPM import drude_dielectric

def sc_mpm():
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

    eps_p = drude_dielectric(k,gamma=0.05,omega_p=1,eps_inf=2)
    mpm = MPM(box,eps_p,guess_type="derivative",tol=1e-3,quiet=True)
    mpm.compute(positions)
    cap,dip= mpm.get_cap_dip()
    return k,cap,dip


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt 

    start = time.time()

    k,C,p = sc_mpm()

    dt = time.time() - start
    print(f"pyMPM class ran in {dt} seconds")

    np.save("cap.npy",C)
    np.save("dip.npy",p)

    plt.plot(k,np.imag(C[:,0,0]),'b-')
    plt.show()
