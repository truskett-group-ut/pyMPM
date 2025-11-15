import numpy as np
from pyMPM import Stratified_Media

def sc_stratified_media():
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
    r = 1 

    etaA = 0.73 #assumed (defintiely not for this example though)
    etaV = 2/3*etaA #for a monolayer
    rho = etaV/(4/3*np.pi*r**3)

    eps_m = 1
    
    eps_sc = eps_m*(1 + rho*C[:,np.identity(3,dtype=bool)])

    eps_sil = 11.7
    h_sil = 500e-6

    h = [2*r,h_sil]
    eps = [eps_sc,eps_sil]
    SM = Stratified_Media(k,h,eps=eps,eps_pre=1,eps_post=1)

    num_theta = 5
    max_theta = 60/180*np.pi
    thetas = np.linspace(0,max_theta,num_theta)
    R,E,T = SM.compute_p(thetas,avg_interference=True,return_media=True)
    return k,R,T


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt 

    start = time.time()

    k,R,T = sc_stratified_media()

    dt = time.time() - start
    print(f"stratified_media ran in {dt} seconds")

    plt.plot(k,R.T)
    plt.show()
    plt.plot(k,T.T)
    plt.show()
