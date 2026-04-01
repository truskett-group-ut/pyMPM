import numpy as np
import time
from ..base_field import Base_Electric_Field

class Dipole_Field(Base_Electric_Field):#{{{
    def __init__(self,box,cutoff,retarded=False,eps_p=None,points=None,dip=None,dip_pos=None,k=None,R=None,debug=False):# {{{
        super().__init__(box=box,eps_p=eps_p,points=points,dip=dip,dip_pos=dip_pos,k=k,R=R,debug=debug)
        if retarded:
            self._real_space = self._full_direct
            self._real_space_precalcs = self._full_dist_precalcs
        else:
            self._real_space = self._quasistatic_direct
            self._real_space_precalcs = self._quasistatic_precalcs
        self.retarded = retarded
        self.rc = cutoff
        # }}}
    def calculate(self):# {{{
        if self.dip_pos is None:
            raise Exception("Dipole positions must be set before calculating electric fields")
        if self.dip is None:
            raise Exception("Dipoles must be set before calculating electric fields")
        if self.has_new_dip_pos or self.has_new_points:
            self._real_space_precalcs()
            self.has_new_points = False
            self.has_new_dip_pos = False
        if self.has_new_eps_p:
            self._calc_alpha()
            self.has_new_eps_p = False
            self.has_new_R = False
        if self.retarded and self.has_new_k:
            self._full_k_precalcs()
            self.has_new_k = False
        return self._electric_field()
    # }}}
    def _electric_field(self):#{{{
        t0 = time.time()
        dipoles = self.dip
        E_point= self.E_point 
        E_point[...] = 0
        E_point = self._real_space(E_point,dipoles)
        self._debug("ef_time",time.time()-t0)
        return E_point
        # }}}

    def _quasistatic_direct(self,E_point,dip):# {{{
        #Real
        p1 = self.p1
        p2 = self.p2
        r = self.r
        d3_inv = self.d3_inv

        if self.calc_inter_dipole:
            alpha_inv = self.alpha_inv
            E_point += dip*alpha_inv[:,None]

        P = dip[p2]
        r_P = np.sum(r*P,axis = -1)
        np.add.at(E_point,p1,(P - 3*r*r_P[:,None])*d3_inv[:,None])
        return E_point
    # }}}
    def _quasistatic_precalcs(self):# {{{
        points = self.points
        dip_pos = self.dip_pos

        p1,p2 = self._gen_neighbor_list()
        r = points[p1] - dip_pos[p2]
        r = r-self.box*(2*r/self.box).astype(int)
        d = np.sqrt(np.sum(r**2,axis = -1))
    
        cutoff_flags = d<self.rc
        d = d[cutoff_flags]
        r = r[cutoff_flags]
        r = r[:,:]/d[:,None]
        d3_inv = d**-3

        self.r = r
        self.d3_inv = d**-3
        self.p1 = p1[cutoff_flags]
        self.p2 = p2[cutoff_flags]
        self.E_point = np.zeros_like(points,dtype=np.complex128)
        # }}}
    def _full_direct(self,E_point,dip):# {{{
        #Real
        p1 = self.p1
        p2 = self.p2
        r = self.r
        d3_inv = self.d3_inv

        if self.calc_inter_dipole:
            alpha_inv = self.alpha_inv
            E_point += dip*alpha_inv[:,None]

        P = dip[p2]
        r_P = np.sum(r*P,axis = -1)
        A = (3*r*r_P[:,None]-P)*self.dot_prefactor
        B = self.k2r * np.cross(np.cross(r,P,-1,-1),r,-1,-1)
        np.add.at(E_point,p1,self.exp*(A+B))
        return E_point
    # }}}
    def _full_dist_precalcs(self):# {{{
        points = self.points
        dip_pos = self.dip_pos

        p1,p2 = self._gen_neighbor_list()
        r = points[p1] - dip_pos[p2]
        r = r-self.box*(2*r/self.box).astype(int)
        d = np.sqrt(np.sum(r**2,axis = -1))
    
        cutoff_flags = d<self.rc
        d = d[cutoff_flags]
        r = r[cutoff_flags]
        r = r[:,:]/d[:,None]
        d3_inv = d**-3


        self.r = r
        self.d = d
        self.d3_inv = d**-3
        self.p1 = p1[cutoff_flags]
        self.p2 = p2[cutoff_flags]
        self.E_point = np.zeros_like(points,dtype=np.complex128)
        # }}}
    def _full_k_precalcs(self):# {{{
        r = self.d
        k = self.k
        ikr2 = 1j*k/r**2
        self.dot_prefactor = (self.d3_inv - ikr2)[:,None]
        self.exp = -np.exp(1j*k*r)[:,None]
        self.k2r = (k**2/r)[:,None]
    # }}}
    def _calc_alpha(self):# {{{
        if self.R is None:
            self.R = 1
        eps = self.eps_p
        self.alpha_inv = (eps+2)/(eps-1)/self.R**3
        # }}}
# }}}
