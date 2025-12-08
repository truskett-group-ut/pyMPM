import numpy as np
import time
import itertools
import warnings
from scipy.special import jv as besselj
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.sparse.linalg import gmres,LinearOperator
from .electric_field import Electric_Field

class MPM():#{{{
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

    def __init__(self,box,eps_p,radius=1,eps_m=1,xi=0.5,tol=1e-3,quiet=False,guess_type="derivative"):# {{{
        self.indent_level = 0
        self.avg_dips = []
        self.dips = []
        self.quiet = quiet

        self.guess_type = guess_type 

        self.xi = xi
        self.errortol = tol
        self.num_particles = None

        self.eps_p = np.copy(eps_p)

        self.box = np.copy(box)
        self.radius = radius
        self.eps_m = eps_m
        self.num_frames = 0
        # }}}
    def compute(self,positions):# {{{
        '''
        Calculate the dipoles for the positions given. May be called multiple times.

        **Parameters**
            *positions*
                An array of shape (num_frames,num_particles,3) or (num_particles,3).

        '''
        positions = np.copy(positions)
        if len(positions.shape) == 2:
            positions = positions[None,...]
        if len(positions.shape) != 3 or positions.shape[-1] != 3:
            raise Exception("MPM.compute(positions) only takes particle position arrays of shape (num_particles,3) and (num_frames,num_particles,3)")

        if self.num_particles is None:
            self._set_dims(positions.shape[1])
            self._nondimensionalize()
            self._precalculations()
            self._set_guess_calc()
            self.EF = Electric_Field(self.box,self.xi,self.errortol,calc_inter_dipole=True)

        elif self.num_particles != positions.shape[1]:
            raise Exception("The number of particles has changed!")

        num_frames = positions.shape[0]
        self.num_frames += num_frames
        if self.num_frames != 1:
            self._print("frame")
            self._increase_indent_level()
        positions = positions / self.length_scale
        for frame_idx in range(num_frames):
            if self.num_frames != 1:
                self._print(f"{frame_idx} of {self.num_frames}")
                self._increase_indent_level()

            self.EF.set_dip_pos(positions[frame_idx])
            self.EF.set_points(positions[frame_idx])

            cap,dip = self._compute_spectrum()

            if self.num_frames != 1:
                self._decrease_indent_level()

            self.avg_dips.append(cap)
            self.dips.append(dip)
        # }}}
    def get_eff_polarizability(self):# {{{
        '''
        Returns the effective polarizability of all particle by averaging all particle dipoles over all frames

        **Returns**
            *alpha_eff*
                Average polarizability with shape (num_wavelengths, 3, 3) in units of eps_m*|E_o|*r^3
        '''
        coef = 1
        return coef*np.squeeze(np.average(self.avg_dips,axis = 0))# }}}
    def get_dipoles(self,):# {{{
        '''
        Returns the average polarizability of all frames considered and the dipoles calulated for all frames

        **Returns**
            *p*
                Dipoles of each particles with shape (num_frames, num_wavelegnths, num_particles, 3, 3) in units of eps_m*|E_o|*R^3.
                Any axis of length one is squeezed out.
        '''
        coef = 1
        return coef*np.squeeze(self.dips)# }}}
    def get_cap_dip(self,):# {{{
        '''
        Deprecated. Use get_dipoles and get_eff_polarizibility instead.
        '''
        return self.get_eff_polarizability(),self.get_dipoles()
    # }}}
    def _compute_spectrum(self):# {{{
        cap = np.zeros([self.num_wavevectors,3,3],dtype = np.complex128)
        dip = np.zeros([self.num_wavevectors,self.num_particles,3,3], dtype = np.complex128)
    
        dip_guess = np.zeros([self.num_particles,3,3]).astype('complex128')
        self._print(f"Wavenumber:")
        self._increase_indent_level()
        for wavevec_idx in range(self.num_wavevectors):
            self._print(f"{wavevec_idx} of {self.num_wavevectors}")

            self_coef = -3/(4*np.pi*(1-self.eps_p[wavevec_idx][:,None]))
            self.EF.set_self_coef(self_coef)

            dip_guess = self._calc_guess(dip_guess,wavevec_idx,dip)
            new_cap, new_dip = self._compute_tensor(dip_guess)
    
            cap[wavevec_idx,:,:] = new_cap
            dip[wavevec_idx,:,:,:] = new_dip
    
        return cap, dip# }}}
    def _compute_tensor(self,dip_guess):# {{{
        E = np.identity(3)
        cap = np.zeros([3,3],dtype = np.complex128)
        dip = np.zeros([self.num_particles,3,3],dtype = np.complex128)
        for dim in range(3):
            dip[:,dim,:] = self._compute_dipoles(E[dim],dip_guess[:,dim])
            cap[dim,:] = np.average(dip[:,dim,:],axis = 0)
        return cap,dip
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
    
    
        E_match = np.array(E.tolist()*num_particles,dtype=np.complex128)
        solve = LinearOperator(2*[3*num_particles], matvec = solve,dtype = np.complex128)
    
        restart = min([num_particles*3,10])
        maxiter = min([num_particles*3,100])

        dip,info = gmres(solve,E_match,x0 = dip_guess,rtol=self.errortol,
                         restart = restart, maxiter = maxiter)
        dip = dip.reshape(num_particles,3)
    
        return dip
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
        dip_guess[:,np.identity(3,dtype=bool)] = (4*np.pi*beta/(1-beta*self.vol_frac))[:,None]
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
            run = (eps_p[im1]-eps_p[im2])[:,None,None]
            new_run = (eps_p[i]-eps_p[im1])[:,None,None]
            zero_flag = run == 0
            run[zero_flag] = 1
            new_run[zero_flag] = 0

            dip_guess[...] = dip[im1] + new_run*rise/run
            
            return dip_guess# }}}
    #}}}
# }}}
