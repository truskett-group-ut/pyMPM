import numpy as np
import scipy.special

#class Ebeam_Field:
    #def __init__(self,eps,):
def Ebeam_Field(pos0,pos,omega,eps,v,a):
    #returns E in units of e/a*c*eps
    pos0 = pos0/a
    pos_ = pos/a 
    pos_shape = pos.shape
    pos_[...,:2] = pos_[...,:2] - pos0[*[None for _ in pos.shape[:-1]],:]
    xy_ = pos_[None,...,:2]
    z_ = pos_[...,2][None,...,None]
    add_omega_axis=False
    if not np.iterable(omega):
        add_omega_axis=True
        omega = np.array([omega])
    num_omega = len(omega)
    omega_ = a*omega[:,*[None for _ in pos_shape]]

    r_ = np.linalg.norm(xy_,axis=-1)[...,None]
    rhat = xy_/r_
    v_ = v
    gamma = (1-eps*v_**2)**-0.5

    prefactor = 4*np.pi*omega_/v_**2/gamma * np.exp(1j*2*np.pi*omega_*z_/v_)
    
    E = np.zeros([num_omega,*pos_shape],dtype=np.complex128)
    xi = 2*np.pi*omega_*r_/v_/gamma

    E[...,2] = (prefactor*1j/gamma*scipy.special.kv(0,xi))[...,0]
    E[...,:2] = prefactor*scipy.special.kv(1,xi) * rhat

    if add_omega_axis:
        return E[0]
    else:
        return E

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    N = 10
    x = np.linspace(-10,10,N)
    y = np.linspace(-10,10,N)
    xy = np.array(np.meshgrid(x,y))
    
    
    pos = np.array([xy[0],xy[1],np.zeros_like(xy[0])]).T
    pos0 = np.array([1,0])
    pos = pos.reshape(N**2,3)
    flag = np.all((pos[:,:2]<1.5) & (pos[:,:2]>-1.5),axis=1)
    pos = pos[~flag]
    xy = xy.T
    
    
    
    omega = np.array([0.002,0.003])
    E = Ebeam_Field(pos0,pos,omega=omega,eps=2,v=0.45,a=1)

    plt.figure(figsize=(4,4))
    for i in range(len(omega)):
        plt.plot(*pos0,"ro")
        plt.quiver(*pos[:,:2].T,*E[i,:,:2].T)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.show()
    
    #E = np.log(np.linalg.norm(np.real(E[...,:2]),axis=-1))
    #plt.pcolormesh(xy[...,0]+240,xy[...,1],E,vmin=-5,vmax=2,)
    #plt.colorbar()
    
    #x = np.linspace(0,1)
    #plt.plot(x,scipy.special.kv(1,x))
    #plt.plot(x,np.imag(scipy.special.kv(1,1j*x)))
    #plt.show()

