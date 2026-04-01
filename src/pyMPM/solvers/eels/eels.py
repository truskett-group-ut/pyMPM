import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres,LinearOperator,bicgstab
from scipy.integrate import simpson
from ...electric_field import Electric_Field
from ...ebeam_field import Ebeam_Field
from ..base_solver import Base_Solver

class EELS(Base_Solver):#{{{
    '''
    Calculates the dipoles of particles for an electric field in x, y, and z direction given particles positions and their dielectric functions.

    **Parameters**
        *box*
            A 1D array of length three specifying *Lx*, *Ly*, and *Lz*.
        *eps_p*
            Either a 1D or 2D array specifying the frequency dependent dielectric function of the particles.
            If *eps_p* is a 1D array, it is assumed to be of length *num_wavelengths* and to specify the frequency dependent dielectric function for all particles. 
            If *eps_p* is 2D, it is assumed to be of shape *num_wavelengths* x *num_particles* and to specify the frequency dependent dielectric function of each particle individually.
        *radius*
            The radius of the particles. Defaults to 1 (assumed nondimentionalized).
        *eps_m*
            The dielectrix constant of the media the particle is embedded in. Defaults to 1 (assumed nondimensionalized).
        *xi*
            The Ewald parameter. Defaults to 0.5.
        *tol*
            The tolerance for solving the dipoles. Defaults to 0.001.
        *quiet*
            If True, does not print out progress statements.
        *guess_type*
            The guess value for each dipole:
                "mean-field"
                    Uses the mean-field approximated dipole as the guess value.
                "previous"
                    Uses the *mean-field* method for the first frequency, then uses the previous-frequency dipole value as the guess value.
                "derivative"
                    Uses the *mean-field* method for the first two frequencies, then uses the last two calculated dipole values to linearly extrapolate the next guess value.
    '''

    def __init__(self,box,eps_p,omega,v,eps_m,radius=1,xi=0.5,tol=1e-3,cutoff=20,# {{{
                      ef_method="ewald",solver_method="bicgstab",guess_type="derivative",retarded=False,
                      split_dist=3,split_frac=0.2,quiet=False,debug=False):

        super().__init__(box,eps_p,eps_m,radius,xi,tol,cutoff,ef_method,guess_type,
                         retarded,solver_method,debug,quiet,omega)
        self.split_dist = split_dist
        self.split_frac = split_frac

        self.v = v
        # }}}
    def compute(self,e_positions,positions,dipoles = None):# {{{
        '''
        Calculate the dipoles for the positions given. May be called multiple times.

        **Parameters**
            *positions*
                An array of shape (num_frames,num_particles,3) or (num_particles,3).

        '''
        positions,e_positions,o_pos_shape,o_epos_shape = self._format_inputs(positions,e_positions)
        user_dips = dipoles

        if self.num_particles is None:
            self._setup(positions.shape[0])
        elif self.num_particles != positions.shape[1]:
            raise Exception("The number of particles has changed!")

        num_epos = e_positions.shape[0]
        if num_epos != 1:
            self._print("epos")

        positions = positions / self.length_scale
        e_positions = e_positions / self.length_scale

        dips = []
        eels = []
        E_fields = []
        poss = []
        for epos_idx in range(num_epos):
            if num_epos != 1:
                self._print(f"{epos_idx} of {num_epos}")
                self._increase_indent_level()

            if user_dips is not None:
                u_dip = user_dips[epos_idx] 
            else:
                u_dip = None 

            e_pos = e_positions[epos_idx]
            pos = np.copy(positions)
            
            pos,R,eps_p = self._split_close_dipoles(pos,e_pos)
            #R = self.radius
            #eps_p = self.eps_p

            self.EF.set_dip_pos(pos)
            self.EF.set_points(pos)
            self.EF.set_R(R)
            self.eels_EF.set_dip_pos(pos)
            self.eels_EF.set_R(R)

            self._precomp_eels(e_pos,pos,R)

            eel,dip,E_field = self._compute_spectrum(e_pos,pos,eps_p,user_dips=u_dip)

            eels.append(eel)
            dips.append(dip)
            poss.append(pos)
            E_fields.append(E_field)

            if num_epos != 1:
                self._decrease_indent_level()

        eels = np.array(eels)
        eels = eels.reshape(*o_epos_shape[:-1],self.num_wavevectors)
        #dips = np.array(dips)
        #dips = dips.reshape(*o_epos_shape[:-1],self.num_wavevectors,-1,3)
        #E_fields = np.array(E_field)
        return eels,dips,poss,E_fields
        # }}}
    def _compute_spectrum(self,epos,positions,eps_p,user_dips):# {{{
        num_particles = len(positions)
        eels = np.zeros([self.num_wavevectors],dtype = np.float64)
        dip = np.zeros([self.num_wavevectors,num_particles,3], dtype = np.complex128)
    
        dip_guess = np.zeros([num_particles,3]).astype('complex128')
        self._print(f"Wavenumber:")

        E = Ebeam_Field(epos,positions,self.omega,self.eps_scale,self.v,1)
        for wavevec_idx in range(self.num_wavevectors):
            self._print(f"{wavevec_idx} of {self.num_wavevectors}")


            if user_dips is None:
                t0 = time.time()
                self.EF.set_eps_p(eps_p[wavevec_idx])
                if self.retarded:
                    self.EF.set_k(2*np.pi*self.omega[wavevec_idx])

                dip_guess = self._calc_guess(dip_guess,eps_p,wavevec_idx,dip)
                if wavevec_idx == 0:
                    Ew = E[wavevec_idx]
                    Ew = Ew/np.linalg.norm(Ew,axis=-1)[:,None]
                    dip_guess = 0.1*dip_guess*Ew * np.sign(Ew)
                    dip_guess[:,2] *= 1j

                Enorm = np.max(np.linalg.norm(np.abs(E[wavevec_idx]),axis=-1))
                dip_guess /= Enorm
                Ei = E[wavevec_idx] / Enorm
                new_dip = self._compute_dipoles(Ei,dip_guess)
                self._debug("guess","Norm(Δp):",np.linalg.norm(new_dip-dip_guess))

                p = new_dip
                p_norm = np.max(np.linalg.norm(np.abs(p),axis=-1))
                p = np.real(p)/p_norm

                self._debug("time","Dipole_Time:",time.time()-t0)

            else:
                new_dip = user_dips[wavevec_idx]
                Enorm = 1

            t0 = time.time()
            new_eels = self._compute_eels(new_dip,self.omega[wavevec_idx])
            self._debug("time","EELS_Time:",time.time()-t0)

            dip[wavevec_idx] = new_dip * Enorm
            eels[wavevec_idx] = new_eels * Enorm
        return eels, dip, E
    # }}}
    def _compute_dipoles(self,E,dip_guess):# {{{
        num_particles = dip_guess.shape[0]

        dip_guess = dip_guess.flatten()
        #---- Preallocations ----
        def solve(dipoles):
            dipoles = dipoles.reshape(num_particles,3)
            self.EF.set_dipoles(dipoles)
            E = self.EF.calculate()
            ret = E.flatten()
            return ret
    
        E_match = E.flatten()

        dip,info = self._solver(solve,E_match,dip_guess)
        if info != 0:
            self._debug("solver","Failed to solve",info)

        dip = dip.reshape(num_particles,3)
        E_match = E_match.reshape(num_particles,3)
    
        return dip
        # }}}
    def _compute_eels(self,dip,omega):# {{{
        self.eels_EF.set_k(omega)
        self.eels_EF.set_dipoles(dip)
        Eind = -self.eels_EF.calculate()
        integrand = np.real(Eind[:,2]*np.exp(-1j*2*np.pi*omega*self.Z_pts/self.v))
        integral = simpson(integrand,self.z_pts)

        #A = np.exp(-1j*2*np.pi*omega*self.Z_pts/self.v)
        #plt.plot(self.z_pts,A.real)
        #plt.plot(self.z_pts,A.imag)
        #plt.show()
        #plt.plot(self.z_pts,integrand)
        #plt.show()
        return integral/(2*np.pi**2*omega)

    # }}}
    def _precomp_eels(self,epos,positions,radius):# {{{
        dz = 0.01
        if np.any(positions <= 0):
            zmin = -self.box[2]/2
            zmax = self.box[2]/2 - dz
            Z = 0
        else:
            zmin = 0
            zmax = self.box[2] - dz
            Z = 1
        P0 = positions[0,2]
        z_pts = np.arange(zmin,zmax+dz,dz)

        #xy = positions[:,:2] - epos[None,:]
        #r = np.linalg.norm(xy,axis=-1)

        #epos = np.tile(epos[None,:],(len(z_pts),1))
        #pts = np.hstack([epos,z_pts[:,None]])
        #self.eels_EF.set_points(pts)

        #dip = np.zeros_like(positions,dtype=np.complex128)
        #dip[:,:2] = (radius**3/r**3)[:,None]*(xy/r[:,None])
        #dip[:,2] = 1j * radius**3/r**3 
        #self.eels_EF.set_dipoles(dip)
        #Eind = -self.eels_EF.calculate()
        #integrand = np.real(Eind[:,2]*np.exp(-1j*2*np.pi*omega*z_pts/self.v))
        #flag = np.isclose(Eind[:,2],0,atol=1e-3)
        #z_pts= z_pts[~flag]
        #print(zmin,np.min(z_pts))
        #print(zmax,np.max(z_pts))
        #raise SystemExit

        #A = np.logspace(-3,np.log10(np.abs(zmax-0.001-P0)),2000)
        #B = np.logspace(-3,np.log10(np.abs(zmin-P0)),2000)
        #z_pts = np.concatenate([P0-np.flip(B),P0+A])

        #mn = -3
        #A = np.logspace(mn,np.log10(self.box[2]/2),5000)
        #z_pts = np.concatenate([[0],P0-np.flip(A),P0+A]) % self.box[2]
        #z_pts = np.sort(z_pts)
        #if np.any(positions < 0):
            #z_pts -= self.box[2]/2

        Z_pts = z_pts  -  P0 + Z*self.box[2]/2
        Z_pts[Z_pts<zmin] = Z_pts[Z_pts<zmin] + self.box[2]
        Z_pts[Z_pts>zmax] = Z_pts[Z_pts>zmax] - self.box[2]
        Z_pts = Z_pts + P0 - Z*self.box[2]/2

        epos = np.tile(epos[None,:],(len(z_pts),1))
        pts = np.hstack([epos,z_pts[:,None]])

        self.eels_EF.set_points(pts)
        self.z_pts = z_pts
        self.Z_pts = Z_pts
        # }}}
    def _setup(self,num_p):# {{{
        super()._setup(num_p)
        if self.ef_method == "ewald":
            self.eels_EF = Electric_Field(self.box,self.xi,self.errortol,method=self.ef_method,debug=self.debug)
        elif self.ef_method == "direct":
            self.eels_EF = Electric_Field(self.box,self.cutoff,method=self.ef_method,retarded=self.retarded,debug=self.debug)
    # }}}
    def _format_inputs(self,positions,e_positions):# {{{
        positions = np.copy(positions)
        if positions.shape[-1] != 3:
            raise Exception("positions must have 3 dimensional coordinates in the last axis")
        o_pos_shape = positions.shape
        if len(positions.shape) > 2:
            positions = positions.reshape(-1,3)
        if e_positions.shape[-1] != 2:
            raise Exception("e_positions must have a 2 dimensional coordinates in the last axis")

        o_epos_shape = e_positions.shape
        if len(e_positions.shape) == 1:
            e_positions = np.array([e_positions])
        elif len(e_positions.shape) > 2:
            e_positions = e_positions.reshape(-1,2)
        return positions,e_positions,o_pos_shape,o_epos_shape# }}}
    def _split_close_dipoles(self,pos,e_pos):# {{{
        x = pos[:,:2] - e_pos[None,:]
        flag = np.linalg.norm(x,axis=-1) < self.split_dist
        r_pos = pos[flag]
        r_R = self.radius[flag]
        r_eps_p = self.eps_p[:,flag]
        pos = pos[~flag]
        new_pos = []
        new_R = []
        new_eps_p = [] 
        N = 0
        #def fibonacci_sphere(R,samples):

            #points = []
            #phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

            #for i in range(samples):
                #y = R*(1 - (i / float(samples - 1)) * 2)  # y goes from 1 to -1
                #radius = np.sqrt(R**2 - y * y)  # radius at y

                #theta = phi * i  # golden angle increment

                #x = np.cos(theta) * radius
                #z = np.sin(theta) * radius

                #points.append((x, y, z))

            #return np.array(points)
        for i in range(len(r_pos)):
            R = r_R[i]
            x = r_pos[i]
            eps_p = r_eps_p[:,i]
            d = R*self.split_frac
            a = np.arange(-R,R+d,d)
            a = (a[1:] + a[:-1])/2
            a = np.array(np.meshgrid(a,a,a)).T
            #a = np.array(np.meshgrid(a,a)).T
            r_flag = np.linalg.norm(a,axis=-1) < R
            a = a[r_flag]
            #a = np.concatenate([a[r_flag],np.zeros_like(a[r_flag,0])[:,None]],axis=-1)
            #a = fibonacci_sphere(R,120)
            new_pos.append(x+a)
            n = a.shape[0]
            N += n
            V = 4/3*np.pi*R**3 / n
            Ri = (3/4/np.pi * V)**(1/3)
            #Ri = d/2
            new_R.append(Ri+np.zeros(n))
            new_eps_p.append(eps_p[:,None]*np.ones(n)[None,:])

        self._print(f"Split {len(r_R)} particle(s) into {N} dipoles")

        #new_pos = np.array(new_pos)[0]
        #zs = np.unique(new_pos[:,2])
        #for z in zs:
            #print(z)
            #iflag = z == new_pos[:,2]
            #plt.scatter(*new_pos[iflag,:2].T)
            #plt.show()

        pos = np.concatenate([pos,*new_pos],axis=0)
        np.savetxt("pos.txt",pos)
        R = self.radius[~flag]
        R = np.concatenate([R,*new_R],axis=0)
        eps_p = self.eps_p[:,~flag]
        eps_p = np.concatenate([eps_p,*new_eps_p],axis=1)
        return pos,R,eps_p
        # }}}
# }}}
