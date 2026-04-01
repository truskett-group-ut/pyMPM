import time
import numpy as np
from scipy.sparse.linalg import gmres,LinearOperator,bicgstab
from scipy.integrate import simpson,quad
from ..electric_field import Electric_Field

class Base_Solver():#{{{
    def __init__(self,box,eps_p,eps_m,radius,xi,tol,cutoff,ef_method,guess_type,retarded,solver_method,debug,quiet,omega):# {{{
        self.indent_level = 0
        self.avg_dips = []
        self.dips = []
        self.ef_method = ef_method
        self.solver_method = solver_method
        self.retarded = retarded

        if isinstance(quiet,bool):
            if quiet:
                self.quiet = -np.inf
            else:
                self.quiet = np.inf
        else:
            self.quiet = quiet

        if isinstance(debug,bool):
            if debug:
                self.debug = []
            else:
                self.debug = []
        elif isinstance(debug,list):
            self.debug = debug
        else:
            self.debug = [debug]

        for i in range(len(self.debug)):
            self.debug[i] = self.debug[i].lower()
            
        self.guess_type = guess_type 

        self.xi = xi
        self.errortol = tol
        self.cutoff = cutoff
        self.num_particles = None

        self.eps_p = np.copy(eps_p)

        self.box = np.copy(box)
        self.radius = radius
        self.eps_m = eps_m
        self.num_frames = 0

        if omega is not None:
            self.omega = omega
        # }}}
    def compute(self,*args,**kwargs):# {{{
        raise NotImplementedError(f"The 'compute' method is not implemented in the {self.__class__.__name__} class")
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
        if hasattr(self,"omega"):
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
    def _setup(self,num_p):# {{{
       self._set_dims(num_p)
       self._nondimensionalize()
       self._precalculations()
       self._set_guess_calc()
       self._set_solver()
       self._set_electric_field()
       #}}}

    #---- Print/Debug Functions ----
    def _debug(self,category,*args):# {{{
        if category not in self.debug:
            return
        print(self.indent_level*"    ",category.upper(),*args)
        return
        # }}}
    def _print(self,*args):# {{{
        if self.indent_level > self.quiet:
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
    #---- Guess Functions ----
    def _set_electric_field(self):# {{{
        if self.ef_method == "ewald":
            self.EF = Electric_Field(self.box,self.xi,self.errortol,method=self.ef_method,debug=self.debug)
        elif self.ef_method == "direct":
            self.EF = Electric_Field(self.box,self.cutoff,method=self.ef_method,retarded=self.retarded,debug=self.debug)
    # }}}
    def _set_solver(self,):# {{{
        shape = 2*[3*self.num_particles]
        restart = min([self.num_particles*3,10])
        maxiter = min([self.num_particles*3,100])

        if self.solver_method == "bicgstab":
            def solver(solve,E_match,dip_guess):
                solve = LinearOperator(shape,matvec=solve,dtype=np.complex128)
                return bicgstab(solve,E_match,dip_guess,rtol=self.errortol,maxiter=restart*maxiter)
            self._solver = solver
        elif self.solver_method == "gmres":
            def solver(solve,E_match,dip_guess):
                solve = LinearOperator(shape,matvec=solve,dtype=np.complex128)
                return gmres(solve,E_match,dip_guess,rtol=self.errortol,maxiter=maxiter,restart=restart)
            self._solver = solver
        else:
            raise NotImplementedError(f"Solver method {self.solver_method} is not implemented")
        # }}}
    def _set_guess_calc(self):# {{{
        if self.guess_type == "mean_field":
            self._calc_guess = self._calc_mean_field_guess
        elif self.guess_type == "previous":
            self._calc_guess = self._calc_previous_guess
        elif self.guess_type == "derivative":
            self._calc_guess = self._calc_derivative_guess
        else:
            raise Exception(f"Guess type {self.guess_type} not supported.")# }}}
    def _calc_mean_field_guess(self,dip_guess,eps_p,wavevec_idx,dip):# {{{
        match_guess = [None for i in dip_guess.shape[1:]]
        beta = (eps_p[wavevec_idx]-1)/(eps_p[wavevec_idx] + 2)
        dip_guess[:,:] = (4*np.pi*beta/(1-beta*self.vol_frac))[:,*match_guess]
        return dip_guess# }}}
    def _calc_previous_guess(self,dip_guess,eps_p,wavevec_idx,dip):# {{{
        if wavevec_idx < 1:
            return self._calc_mean_field_guess(dip_guess,eps_p,wavevec_idx,dip)
        dip_guess[...] = dip[wavevec_idx-1]
        return dip_guess
    # }}}
    def _calc_derivative_guess(self,dip_guess,eps_p,wavevec_idx,dip):# {{{
        if wavevec_idx < 2:
            return self._calc_previous_guess(dip_guess,eps_p,wavevec_idx,dip)
        else:
            match_guess = [None for i in dip_guess.shape[1:]]
            i = wavevec_idx
            im3 = i - 3
            im2 = i - 2
            im1 = i - 1

            alpha = (eps_p-1)/(eps_p+2)
            X = alpha**-1
            #X = eps_p
            #X = self.omega[:,None]

            y1 = dip[im1]
            dy1 = (dip[im1]-dip[im2])
            dx1 = (X[im1]-X[im2])[:,*match_guess]
            dydx1 = dy1/dx1
            dx0 = (X[i]-X[im1])[:,*match_guess]


            #if i > 2:
                #dy2 = (dip[im2]-dip[im3])
                #dx2 = (X[im2]-X[im3])[:,None]
                #dydx2 = dy2/dx2
                #d2ydx2 = (dydx1-dydx2) / dx1
                #dydx0 = dydx1 + dx0*d2ydx2
                #y0 =  y1 + dx0*dydx0
            #else:
            y0 = y1 + dx0*dydx1

            dip_guess[...] = y0
            
            return dip_guess
        # }}}
# }}}
