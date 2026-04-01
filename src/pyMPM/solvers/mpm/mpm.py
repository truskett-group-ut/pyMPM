import time
import numpy as np
from ..base_solver import Base_Solver

class MPM(Base_Solver):#{{{
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

    def __init__(self,box,eps_p,eps_m=1,radius=1,xi=0.5,tol=1e-3,cutoff=10,ef_method="ewald",retarded=False,solver_method="gmres",guess_type="derivative",quiet=False,debug=False,omega=None,E0 = None):# {{{
        super().__init__(box,eps_p,eps_m,radius,xi,tol,cutoff,ef_method,guess_type,retarded,solver_method,debug,quiet,omega)
        self.avg_dips = []
        self.dips = []
        self.E0 = E0
        # }}}
    def compute(self,positions):# {{{
        '''
        Calculate the dipoles for the positions given. May be called multiple times.

        **Parameters**
            *positions*
                An array of shape (num_frames,num_particles,3) or (num_particles,3).

        '''
        positions = self._format_inputs(positions)

        if self.num_particles is None:
            self._setup(positions.shape[1])
        elif self.num_particles != positions.shape[1]:
            raise Exception("The number of particles has changed!")

        num_frames = positions.shape[0]
        self.num_frames += num_frames
        if self.num_frames != 1:
            self._print("frame")

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
    def _setup(self,num_p):# {{{
        super()._setup(num_p)
        self._set_E0()
    # }}}
    def _compute_spectrum(self):# {{{
        cap = np.zeros([self.num_wavevectors,*self.E0.shape],dtype = np.complex128)
        dip = np.zeros([self.num_wavevectors,self.num_particles,*self.E0.shape], dtype = np.complex128)
    
        dip_guess = np.zeros([self.num_particles,*self.E0.shape]).astype('complex128')
        self._print(f"Wavenumber:")
        for wavevec_idx in range(self.num_wavevectors):
            t0 = time.time()
            self._print(f"{wavevec_idx} of {self.num_wavevectors}")

            self.EF.set_eps_p(self.eps_p[wavevec_idx])
            if self.retarded:
                self.EF.set_k(2*np.pi*self.omega[wavevec_idx])

            dip_guess = self._calc_guess(dip_guess,self.eps_p,wavevec_idx,dip)
            new_cap, new_dip = self._compute_tensor(dip_guess)
    
            cap[wavevec_idx,:,:] = new_cap
            dip[wavevec_idx,:,:,:] = new_dip
            self._debug("time","Dipole_Time:",time.time()-t0)
    
        return cap, dip# }}}
    def _compute_tensor(self,dip_guess):# {{{
        cap = np.zeros(self.E0.shape,dtype = np.complex128)
        dip = np.zeros([self.num_particles,*self.E0.shape],dtype = np.complex128)
        for dim in range(self.E0.shape[0]):
            dip[:,dim,:] = self._compute_dipoles(self.E0[dim],dip_guess[:,dim])
            cap[dim,:] = np.average(dip[:,dim,:],axis = 0)
        return cap,dip
    # }}}
    def _compute_dipoles(self,E,dip_guess):# {{{
        num_particles = self.num_particles
    
        dip_guess = dip_guess.flatten()
    
        def solve(dipoles):
            dipoles = dipoles.reshape(num_particles,3)
            self.EF.set_dipoles(dipoles)
            E = self.EF.calculate()
            ret = E.flatten()
            return ret
    
        E_match = np.array(E.tolist()*num_particles,dtype=np.complex128)
    
        dip,info = self._solver(solve,E_match,dip_guess)
        dip = dip.reshape(num_particles,3)
    
        return dip
        # }}}
    def _format_inputs(self,positions):# {{{
        positions = np.copy(positions)
        if len(positions.shape) == 2:
            positions = positions[None,...]
        if len(positions.shape) != 3 or positions.shape[-1] != 3:
            raise Exception("MPM.compute(positions) only takes particle position arrays of shape (num_particles,3) and (num_frames,num_particles,3)")
        return positions# }}}
    def _set_E0(self):# {{{
        if self.E0 is None:
            self.E0 = np.identity(3)
        else:
            self.E0 = np.asarray(self.E0)
            if len(self.E0.shape) != 2 or self.E0.shape[1] != 3:
                raise Exception(f"Shape of E0 must be [n,3], not {self.E0.shape}")# }}}
# }}}
