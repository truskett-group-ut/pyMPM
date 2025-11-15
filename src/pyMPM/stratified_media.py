import numpy as np

class Stratified_Media():
    '''
    Calculates the far-field spectrum of a stratified media. 

    **Parameters**
        *omega*
            A 1D array (num_wavelengths) specifying the wavenumbers (inverse length) to be calculated.
        *h*
            A 1D array (num_layers) specifying the thickness of each layer.
        *eps* (*n*)
            A 1D array (num_layers) specifying the permitivity (refractive index) of each layer.
            Each dielectric function may be specified as either a scalar (frequency independent), 1D array (num_wavelength; isotropic), or 2D (num_wavelengths,3; anisotropic) array.
            Only *eps* or *n* should be specified.
        *eps_inf* (*eps_pre* / *eps_post*)
            Specify the permitivities of the two semi-infinite media on either side of the stratified media.
    '''
    def __init__(self,omega,h,n=None,n_inf=None,n_pre=None,n_post=None,eps=None,eps_inf=None,eps_pre=None,eps_post=None):# {{{
        self.dims = 3

        h = np.asarray(h)
        self.h = h
        self.num_layers = len(h)
        if len(h.shape) != 1:
            raise Exception("'h' must be a 1-D array of layer heights")

        omega = np.asarray(omega)
        if len(omega.shape) != 1:
            raise Exception("'omega' must be a 1-D array of wavenumbers (of inverse length)")
        self.omega = 2*np.pi*omega
        self.num_waves = len(omega)

        n_passed = np.any([n is not None,n_inf is not None,n_pre is not None,n_post is not None])
        eps_passed = np.any([eps is not None,eps_inf is not None,eps_pre is not None,eps_post is not None])

        if n_passed and eps_passed:
            raise Exception("Pass either refractive indexes (n) or permitivities (eps)")
        if eps_passed:
            n = eps
            n_inf = eps_inf
            n_pre = eps_pre
            n_post= eps_post

        if len(n) != self.num_layers:
            raise Exception("Length of 'h' and 'n' must be the same")

        if n_inf is None and (n_pre is None or n_post is None):
            raise Exception("'n_inf' or 'n_pre' AND 'n_post' must be defined")
        elif n_inf is not None and (n_pre is not None or n_post is not None):
            raise Exception("Only 'n_inf' OR 'n_pre' and 'n_post' can be defined")
        elif n_inf is not None:
            n_pre = n_inf
            n_post = n_inf
        if isinstance(n,np.ndarray):
            n_new = list()
            for ni in n:
                n_new.append(ni)
            n = n_new

        n.insert(0,n_pre)
        n.append(n_post)
        self.num_layers += 2
        self.num_interfaces = self.num_layers-1
        for i in range(self.num_layers):
            ni = n[i]
            if not np.iterable(n[i]):
                n[i] = ni*np.ones([self.num_waves,self.dims])
                continue
            ni = np.asarray(ni)
            if len(ni.shape) == 1:
                if ni.shape[0] == self.num_waves and ni.shape[0] == self.dims:
                    raise Exception("When 'omega' is length 3, a 1-D 'ni' is ambiguous")
                elif ni.shape[0] == self.num_waves:
                    n[i] = np.tile(ni,(self.dims,1)).T
                elif ni.shape[0] == self.dims:
                    n[i] = np.tile(ni,(self.num_waves,1))
                else:
                    raise Exception(f"Cannot justify a 1-D 'ni' of length {ni} with an 'omega' of length {self.num_waves}")
            else:
                if ni.shape[0] != self.num_waves or ni.shape[1] != self.dims:
                    raise Exception(f"A 2-D 'ni' must be shape ({self.num_waves} x {self.dims})")

        n = np.asarray(n)
        if eps_passed:
            n = np.sqrt(n)

        self.n = n.astype(np.complex128)
        self.x = np.array([0,*np.cumsum(h)])
        return
    # }}}
    def compute(self, incident_angle,p_type=None,s_type=None,avg_interference=True,res=None,return_fields=False,return_media=False):# {{{
        if p_type is None and s_type is None:
            p_type = True
            s_type = True

        if p_type:
            PR = []
            PT = []
            if return_media:
                PE = []

        if s_type:
            SR = []
            ST = []
            if return_media:
                SE = []

        if avg_interference:
            if res is None:
                self.res = 100
            else:
                self.res = res
        else:
            self.res = 2

        if not np.iterable(incident_angle):
            incident_angle = [incident_angle]
            idx = np.s_[0]
        else:
            idx = np.s_[:]

        for theta in incident_angle:
            if s_type:
                r,e,t = self._compute_s(theta,avg_interference,return_fields)
                ST.append(t)
                SR.append(r)
                if return_media:
                    SE.append(e)
            if p_type:
                r,e,t = self._compute_p(theta,avg_interference,return_fields)
                PT.append(t)
                PR.append(r)
                if return_media:
                    PE.append(e)

        returns = []
        if return_media:
            if s_type and p_type:
                return [np.squeeze(SR),np.squeeze(SE),np.squeeze(ST)], [np.squeeze(PR),np.squeeze(PE),np.squeeze(PT)]
            elif s_type:
                return np.squeeze(SR),np.squeeze(SE),np.squeeze(ST)
            elif p_type:
                return np.squeeze(PR),np.squeeze(PE),np.squeeze(PT)
        else:
            if s_type and p_type:
                return [np.squeeze(SR),np.squeeze(ST)], [np.squeeze(PR),np.squeeze(PT)]
            elif s_type:
                return np.squeeze(SR),np.squeeze(ST)
            elif p_type:
                return np.squeeze(PR),np.squeeze(PT)
        # }}}
    def compute_p(self,incident_angle,avg_interference=True,res=None,return_fields=False,return_media=False):# {{{
        '''
        Compute the far-field properties for a p-polarized incident beam at *incident-angle*.

        **Parameters**
            *incident_angle*
                A scalar or 1D array of incident angles to calculate far-field spectra for.
            *avg_interference*
                If True, will average over the interference band period to account for incoherence.
            *res*
                The number of points to average over in the interference band period.
                Only used if *avg_interference* is True.
                Defaults to 100.
            *return_fields*
                If True, returns fields instead of powers.
            *return_media*
                If True, will return the values calculated inside the media in addition to the reflectance and transmittance.
        **Returns**
            *R*
                A 2D array (num_angles,num_wavelengths) containing the power of the backwards propagating wave at the first interface, or reflectance. Axes of length one are removed.
            *M*
                A 3D array (num_angle,2*num_layers,num_wavelengths) containing the power of the forward and backwards propagating waves in each layer of the media.
                *M* is only returned if *return_media* is True.
            *T*
                A 2D array (num_angles,num_wavelengths) containing the power of the forward propagating wave at the last interface, or transmittance. Axes of length one are removed.

        '''
        R = []
        T = []
        if return_media:
            E = []

        if avg_interference:
            if res is None:
                self.res = 100
            else:
                self.res = res
        else:
            self.res = 2

        for theta in incident_angle:
            r,e,t = self._compute_p(theta,avg_interference,return_fields)
            T.append(t)
            R.append(r)
            if return_media:
                E.append(e)

        if return_media:
            return np.squeeze(R),np.squeeze(E),np.squeeze(T)
        else:
            return np.squeeze(R),np.squeeze(T)
        # }}}
    def compute_s(self, incident_angle,avg_interference=True,res=None,return_fields=False,return_media=False):# {{{
        '''
        Compute the far-field properties for an s-polarized incident beam at *incident-angle*.

        **Parameters**
            *incident_angle*
                A scalar or 1D array of incident angles to calculate far-field spectra for.
            *avg_interference*
                If True, will average over the interference band period to account for incoherence.
            *res*
                The number of points to average over in the interference band period.
                Only used if *avg_interference* is True.
                Defaults to 100.
            *return_fields*
                If True, returns fields instead of powers.
            *return_media*
                If True, will return the values calculated inside the media in addition to the reflectance and transmittance.
        **Returns**
            *R*
                A 2D array (num_angles,num_wavelengths) containing the power of the backwards propagating wave at the first interface, or reflectance. Axes of length one are removed.
            *M*
                A 3D array (num_angle,2*num_layers,num_wavelengths) containing the power of the forward and backwards propagating waves in each layer of the media.
                *M* is only returned if *return_media* is True.
            *T*
                A 2D array (num_angles,num_wavelengths) containing the power of the forward propagating wave at the last interface, or transmittance. Axes of length one are removed.

        '''
        R = []
        T = []
        if return_media:
            E = []

        if avg_interference:
            if res is None:
                self.res = 100
            else:
                self.res = res
        else:
            self.res = 2

        for theta in incident_angle:
            r,e,t = self._compute_s(theta,avg_interference,return_fields)
            T.append(t)
            R.append(r)
            if return_media:
                E.append(e)

        if return_media:
            return np.squeeze(R),np.squeeze(E),np.squeeze(T)
        else:
            return np.squeeze(R),np.squeeze(T)
        # }}}
    def _compute_s(self,theta0,average_interference,return_fields):# {{{
        # [layers,waves,dim,res]
        E = np.zeros([2*self.num_layers-2,self.num_waves],dtype=np.complex128)

        ny = self.n[:,:,1]
        ny = ny[:,:,None]
        k = ny*self.omega[None,:,None]
        x = self.x[:,None,None]
        theta = np.arcsin(ny[0]/ny*np.sin(theta0))

        if average_interference:
            p = 2*np.pi / (self.x[-1] * np.cos(theta[:,:]))[...,0]
            P = np.transpose(np.linspace(-p/2,p/2,self.res),axes=[1,2,0])
        else:
            p = np.ones([self.num_layers,self.num_waves])*0.001
            P = np.zeros([self.num_layers,self.num_waves,self.res],dtype=np.complex128)
            P[...,1] += 0.001

        i = np.s_[:-1,...]
        i1= np.s_[1:,...]

        a = 1j*(k[i] +P[i])*x*np.cos(theta[i])
        a1 =1j*(k[i1]+P[i1])*x*np.cos(theta[i1])

        j1 = 2*np.arange(self.num_interfaces)
        A = np.zeros([2*self.num_interfaces,2*self.num_layers,self.num_waves,self.res],dtype = np.complex128)
        A[j1,j1+0]   = np.exp( a)
        A[j1,j1+1]   = np.exp(-a)
        A[j1,j1+2]   =-np.exp( a1)
        A[j1,j1+3]   =-np.exp(-a1)
        A[j1+1,j1+0] = ny[i]*np.cos(theta[i])*np.exp(a)
        A[j1+1,j1+1] =-ny[i]*np.cos(theta[i])*np.exp(-a)
        A[j1+1,j1+2] =-ny[i1]*np.cos(theta[i1])*np.exp(a1)
        A[j1+1,j1+3] = ny[i1]*np.cos(theta[i1])*np.exp(-a1)

        A = A[:,1:-1]
        A = A.reshape(2*self.num_interfaces,2*self.num_interfaces,self.num_waves*self.res)
        A = np.transpose(A,axes=(2,0,1))

        b = np.zeros([2*self.num_interfaces,self.num_waves,self.res],dtype=np.complex128)
        b[0] = -1
        b[1] = (-ny*np.cos(theta))[0]
        b = b.reshape(2*self.num_interfaces,self.num_waves*self.res)
        b = np.transpose(b,axes=(1,0))

        Es = np.linalg.solve(A,b)
        Es = np.transpose(Es,axes=(1,0))
        Es = Es.reshape(2*self.num_interfaces,self.num_waves,self.res)

        Z = np.concatenate([[0],
                [i for i in range(1,self.num_layers-1) for j in range(2)],
                [self.num_layers-1]]).astype(int)
        E = np.trapz(Es,P[Z],axis=-1)/p[Z]

        if return_fields:
            return  E[0],E[1:-1],E[-1]

        ny = ny[...,0]
        theta = theta[...,0]
        E = np.abs(E)**2 * (ny[Z] * np.cos(theta[Z])).real / (ny[0] * np.cos(theta[0])).real
        return  E[0],E[1:-1],E[-1]
    # }}}
    def _compute_p(self,theta0,average_interference,return_fields):# {{{
        E = np.zeros([2*self.num_layers-2,self.num_waves],dtype=np.complex128)

        nx = self.n[:,:,0]
        nx2 = nx**2
        nz = self.n[:,:,2]
        nz2 = nz**2

        theta = np.arcsin(nx[0]*nz[0]*nz*np.sin(theta0) /\
                np.sqrt((nx2*nz2*(nx2[0]-nz2[0]) - nx2[0]*nz2[0]*(nx2-nz2))*np.sin(theta0)**2 + nx2*nz2*nz2[0]))
        m = nx*nz / np.sqrt(nx2*np.sin(theta)**2+nz2*np.cos(theta)**2)

        k = m[:,:,None]*self.omega[None,:,None] 
        theta = theta[:,:,None]
        nx2 = nx2[:,:,None]
        nx = nx[:,:,None]
        nz = nz[:,:,None]
        m = m[:,:,None]
        x = self.x[:,None,None] 

        if average_interference:
            p = 2*np.pi / (self.x[-1] * np.cos(theta[:,:]))[...,0]
            P = np.transpose(np.linspace(-p/2,p/2,self.res),axes=[1,2,0])
        else:
            p = np.ones([self.num_layers,self.num_waves])*0.001
            P = np.zeros([self.num_layers,self.num_waves,self.res],dtype=np.complex128)
            P[...,1] += 0.001


        i = np.s_[:-1,...]
        i1= np.s_[1:,...]
        j1 = 2*np.arange(self.num_interfaces)[:,None]
        j2 = j1 + np.arange(4)[None,:]
        a = 1j*(k[i] + P[i])*x*np.cos(theta[i])
        a1= 1j*(k[i1] + P[i1])*x*np.cos(theta[i1])

        A = np.zeros([2*self.num_interfaces,2*self.num_layers,self.num_waves,self.res],dtype = np.complex128)
        A[j1,j2] = np.swapaxes([
                                nx2[i]/m[i]*np.exp(a),
                                nx2[i]/m[i]*np.exp(-a),
                                -nx2[i1]/m[i1]*np.exp(a1),
                                -nx2[i1]/m[i1]*np.exp(-a1),
                               ],0,1)
        A[j1+1,j2] = np.swapaxes([
                                  np.cos(theta[i])*np.exp(a),
                                  -np.cos(theta[i])*np.exp(-a),
                                  -np.cos(theta[i1])*np.exp(a1),
                                  np.cos(theta[i1])*np.exp(-a1),
                                  ],0,1)

        A = A[:,1:-1]
        A = A.reshape(2*self.num_interfaces,2*self.num_interfaces,self.num_waves*self.res)
        A = np.transpose(A,axes=(2,0,1))

        b = np.zeros([2*self.num_interfaces,self.num_waves,self.res],dtype=np.complex128)
        b[0] = (-nx2/m)[0,:,:]
        b[1] = (-np.cos(theta))[0,:,:]
        b = b.reshape(2*self.num_interfaces,self.num_waves*self.res)
        b = np.transpose(b,axes=(1,0))

        Es = np.linalg.solve(A,b)
        Es = np.transpose(Es,axes=(1,0))
        Es = Es.reshape(2*self.num_interfaces,self.num_waves,self.res)

        Z = np.concatenate([[0],
                [i for i in range(1,self.num_layers-1) for j in range(2)],
                [self.num_layers-1]]).astype(int)
        E = np.trapz(Es,P[Z],axis=-1)/p[Z]

        if return_fields:
            return  E[0],E[1:-1],E[-1]

        m = m[...,0]
        theta = theta[...,0]
        E = np.abs(E)**2 * (m[Z] * np.cos(theta[Z])).real / (m[0] * np.cos(theta[0])).real
        return  E[0],E[1:-1],E[-1]
    # }}}
