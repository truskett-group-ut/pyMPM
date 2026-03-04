import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres,LinearOperator
from scipy.integrate import simpson,quad
from .electric_field import Electric_Field
from .ebeam_field import Ebeam_Field

class EELS():#{{{
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

    def __init__(self,box,eps_p,omega,v,eps_m,radius=1,xi=0.5,tol=1e-3,quiet=False,guess_type="derivative"):# {{{
        self.indent_level = 0
        self.avg_dips = []
        self.dips = []
        self.quiet = quiet

        self.guess_type = guess_type 

        self.xi = xi
        self.errortol = tol
        self.num_particles = None

        self.eps_p = np.copy(eps_p)

        self.omega = omega
        self.v = v

        self.box = np.copy(box)
        self.radius = radius
        self.eps_m = eps_m
        self.num_frames = 0
        # }}}
    def compute(self,e_positions,positions,dips = None):# {{{
        '''
        Calculate the dipoles for the positions given. May be called multiple times.

        **Parameters**
            *positions*
                An array of shape (num_frames,num_particles,3) or (num_particles,3).

        '''
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

        if dips is not None:
            self.have_dips = True
            self.dips = dips
        else:
            self.have_dips = False

        if self.num_particles is None:
            self._set_dims(positions.shape[0])
            self._nondimensionalize()
            self._precalculations()
            self._set_guess_calc()

            self.EF = Electric_Field(self.box,self.xi,self.errortol,calc_inter_dipole=True)
            self.eels_EF = Electric_Field(self.box,self.xi,self.errortol,calc_inter_dipole=False)

        elif self.num_particles != positions.shape[1]:
            raise Exception("The number of particles has changed!")

        num_epos = e_positions.shape[0]
        if num_epos != 1:
            self._print("epos")
            self._increase_indent_level()

        positions = positions / self.length_scale
        e_positions = e_positions / self.length_scale

        self.EF.set_dip_pos(positions)
        self.EF.set_points(positions)
        self.eels_EF.set_dip_pos(positions)

        dips = []
        eels = []
        E_fields = []
        for epos_idx in range(num_epos):
            if self.num_frames != 1:
                self._print(f"{epos_idx} of {num_epos}")
                self._increase_indent_level()

            self._precomp_eels(e_positions[epos_idx],positions)
            eel,dip,E_field = self._compute_spectrum(e_positions[epos_idx],positions)

            if self.num_frames != 1:
                self._decrease_indent_level()
            eels.append(eel)
            dips.append(dip)
            E_fields.append(E_field)
        eels = np.array(eels)
        eels = eels.reshape(*o_epos_shape[:-1],self.num_wavevectors)
        dips = np.array(dips)
        dips = dips.reshape(*o_epos_shape[:-1],self.num_wavevectors,*o_pos_shape)
        E_fields = np.array(E_field)
        return eels,dips,E_fields
        # }}}
    def _compute_spectrum(self,epos,positions):# {{{
        eels = np.zeros([self.num_wavevectors],dtype = np.float64)
        dip = np.zeros([self.num_wavevectors,self.num_particles,3], dtype = np.complex128)
    
        dip_guess = np.zeros([self.num_particles,3]).astype('complex128')
        self._print(f"Wavenumber:")
        self._increase_indent_level()

        E = Ebeam_Field(epos,positions,self.omega,self.eps_scale,self.v,1)
        for wavevec_idx in range(self.num_wavevectors):
            self._print(f"{wavevec_idx} of {self.num_wavevectors}")

            if not self.have_dips:
                self_coef = -3/(4*np.pi*(1-self.eps_p[wavevec_idx][:,None]))
                self.EF.set_self_coef(self_coef)

                dip_guess = self._calc_guess(dip_guess,wavevec_idx,dip)
                if wavevec_idx < 2:
                    Ew = E[wavevec_idx]
                    Ew = Ew/np.linalg.norm(Ew,axis=-1)[:,None]
                    dip_guess = dip_guess * Ew

                Enorm = np.max(np.linalg.norm(np.abs(E[wavevec_idx]),axis=-1))
                Ei = E[wavevec_idx] / Enorm
                new_dip = self._compute_dipoles(Ei,dip_guess)
            else:
                new_dip = self.dips[wavevec_idx]
                Enorm = 1

            new_eels = self._compute_eels(new_dip,self.omega[wavevec_idx])

            dip[wavevec_idx] = new_dip * Enorm
            eels[wavevec_idx] = new_eels * Enorm
        return eels, dip, E
    # }}}
    def _compute_dipoles(self,E,dip_guess):# {{{
        num_particles = self.num_particles

        dip_guess = dip_guess.flatten()
        #---- Preallocations ----
        def solve(dipoles):
            dipoles = dipoles.reshape(num_particles,3)
            self.EF.set_dipoles(dipoles)
            E = self.EF.calculate()
            ret = E.flatten()
            return ret
    
        #E_match = np.array(E.tolist()*num_particles,dtype=np.complex128)
        E_match = E.flatten()
        solve = LinearOperator(2*[3*num_particles], matvec = solve,dtype = np.complex128)
    
        restart = min([num_particles*3,10])
        maxiter = min([num_particles*3,100])

        dip,info = gmres(solve, E_match, x0=dip_guess, rtol=self.errortol,
                         restart=restart, maxiter=maxiter)
        dip = dip.reshape(num_particles,3)
        E_match = E_match.reshape(num_particles,3)
    
        #E_solve = solve(dip.flatten())
        #for i in range(len(E_match)):
            #print(E_match[i])
            #print(dip[i])
        #input()

        return dip
        # }}}
    def _compute_eels(self,dip,omega):# {{{
        


        self.eels_EF.set_dipoles(dip)

        #for i in range(3):
            #dipc = np.zeros_like(dip)
            #dipc[...,i] = dip[...,i]
            #self.eels_EF.set_dipoles(dipc)
            #Eind = -self.eels_EF.calculate()
            #integrand = np.real(Eind[:,2]*np.exp(1j*2*np.pi*omega*z_pts/self.v))
            #integral = simpson(integrand,z_pts)
            #print(integral)
            #plt.plot(z_pts,integrand)
            #plt.show()

        #A = np.exp(1j*2*np.pi*omega*Z_pts/self.v)
        #plt.plot(z_pts,A.real)
        #plt.plot(z_pts,A.imag)
        #plt.plot(z_pts,np.abs(A))
        #plt.show()
        Eind = -self.eels_EF.calculate()
        integrand = np.real(Eind[:,2]*np.exp(-1j*2*np.pi*omega*self.Z_pts/self.v))
        integral = -simpson(integrand,self.z_pts)
        #print(integral)
        #plt.plot(z_pts,Eind[:,2].real)
        #plt.plot(z_pts,Eind[:,2].imag)
        #plt.plot(z_pts,integrand)
        #plt.show()
        #plt.plot(z_pts,Eind[:,2].real)
        #plt.plot(z_pts,Eind[:,2].imag)
        return integral/(2*np.pi**2*omega)

    # }}}
    def _set_dims(self,num_p):# {{{
        self.num_particles = num_p

        if not np.iterable(self.radius):
            self.radius = self.radius*np.ones(num_p)
        self.radius = np.asarray(self.radius)
        self.radius = np.copy(self.radius)
        if len(self.radius.shape) != 1:
            raise Exception("radius must be passed as a scalar or 1-D array of length num_particles")
        elif len(self.radius) != num_p:
            raise Exception("The number of particles provided by positions is inconsistent with the number of radii provided")
        elif not np.all(self.radius == self.radius[0]):
            return NotImplementedError("Radii of different sizes not yet supported")

        if not np.iterable(self.eps_p):
            self.eps_p = self.eps_p*np.ones([1,num_p])
        self.eps_p = np.asarray(self.eps_p)
        if len(self.eps_p.shape) == 1:
            self.eps_p = np.repeat(self.eps_p[:,None],num_p,axis=1)
        elif len(self.eps_p.shape) > 2:
            raise Exception("eps_p must be input as either a scalar, 1-D array (length num_waves)," +\
                            " or 2-D array (length num_waves x num_particles)")
        if self.eps_p.shape[1] != num_p:
            raise Exception("The number of particles provided by positions is inconsistent with the eps_p provided")
        self.num_wavevectors = self.eps_p.shape[0]

        # }}}
    def _nondimensionalize(self,):# {{{
        self.length_scale = self.radius[0]
        self.eps_scale = self.eps_m
        self.energy_scale = 1

        self.box = self.box/self.length_scale
        self.radius = self.radius/self.length_scale
        self.omega = self.omega*self.length_scale

        self.eps_m  = self.eps_m/self.eps_scale
        self.eps_p = self.eps_p/self.eps_scale
        # }}}
    def _calc_vol_frac(self):#{{{
        self.vol_frac = 4/3*np.pi * np.sum(self.radius**3) / np.prod(self.box)
        return
        #}}}
    def _precalculations(self):# {{{
        self._calc_vol_frac()
        return 
    # }}}
    def _precomp_eels(self,epos,positions):# {{{
        dz = 0.01
        if np.any(positions < 0):
            zmin = -self.box[2]/2
            zmax = self.box[2]/2 - dz
            Z = 0
        else:
            zmin = 0
            zmax = self.box[2] - dz
            Z = 1
        P0 = positions[0,2]
        z_pts = np.arange(zmin,zmax+dz,dz)

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
    #{{{ ---- Print Functions ----
    def _print(self,*args):# {{{
        if self.quiet:
            return
        print(self.indent_level*"    ", *args)
        return
        # }}}
    def _increase_indent_level(self,):# {{{
        self.indent_level += 1
        return
        # }}}
    def _decrease_indent_level(self,):# {{{
        self.indent_level -= 1
        return
        # }}}
    #}}}
    #{{{ ---- Guess Functions ----
    def _set_guess_calc(self):# {{{
        if self.guess_type == "mean_field":
            self._calc_guess = self._calc_mean_field_guess
        elif self.guess_type == "previous":
            self._calc_guess = self._calc_previous_guess
        elif self.guess_type == "derivative":
            self._calc_guess = self._calc_derivative_guess
        else:
            raise Exception(f"Guess type {self.guess_type} not supported.")# }}}
    def _calc_mean_field_guess(self,dip_guess,wavevec_idx,dip):# {{{
        beta = (self.eps_p[wavevec_idx]-1)/(self.eps_p[wavevec_idx] + 2)
        dip_guess[:,:] = (4*np.pi*beta/(1-beta*self.vol_frac))[:,None]
        return dip_guess# }}}
    def _calc_previous_guess(self,dip_guess,wavevec_idx,dip):# {{{
        if wavevec_idx < 1:
            return self._calc_mean_field_guess(dip_guess,wavevec_idx,dip)
        dip_guess[...] = dip[wavevec_idx-1]
        return dip_guess
    # }}}
    def _calc_derivative_guess(self,dip_guess,wavevec_idx,dip):# {{{
        if wavevec_idx < 2:
            return self._calc_mean_field_guess(dip_guess,wavevec_idx,dip)
        else:
            i = wavevec_idx
            im2 = i - 2
            im1 = i - 1
            eps_p = self.eps_p

            rise = (dip[im1]-dip[im2])
            run = (eps_p[im1]-eps_p[im2])[:,None]
            new_run = (eps_p[i]-eps_p[im1])[:,None]
            zero_flag = run == 0
            run[zero_flag] = 1
            new_run[zero_flag] = 0

            dip_guess[...] = dip[im1] + new_run*rise/run
            
            return dip_guess# }}}
    #}}}
# }}}
