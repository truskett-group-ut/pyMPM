import numpy as np
import time
import itertools
import warnings
from scipy.special import jv as besselj
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.sparse.linalg import gmres,LinearOperator

class Electric_Field():#{{{
    def __init__(self,box,xi,errortol,calc_inter_dipole=False,points = None,dip = None,dip_pos=None):# {{{
        self.box = box
        self.xi = xi
        self.errortol = errortol

        self.dip = dip
        self.dip_pos = dip_pos

        self.calc_inter_dipole = calc_inter_dipole

        self.has_new_dip_pos = True
        self.has_new_points = True

        self._precalculations()
        # }}}
    def set_dipoles(self,dip):# {{{
        self.dip = dip
        # }}}
    def set_dip_pos(self,dip_pos):# {{{
        self.dip_pos = dip_pos
        self.has_new_dip_pos = True
    # }}}
    def set_self_coef(self,self_coef):# {{{
        self.self_coef = self_coef
    # }}}
    def set_points(self,points):# {{{
        self.points = points
        self.has_new_points = True
    # }}}
    def calculate(self):# {{{
        if self.dip_pos is None:
            raise Exception("Dipole positions must be set before calculating electric fields")
        if self.dip is None:
            raise Exception("Dipoles must be set before calculating electric fields")
        if self.has_new_dip_pos or self.has_new_points:
            self._real_space_precalcs()
        if self.has_new_points:
            self._contract_precalcs()
            self.has_new_points = False
        if self.has_new_dip_pos:
            self._spread_precalcs()
            self.has_new_dip_pos = False
        return self._electric_field()
    # }}}
    def _electric_field(self):#{{{
        dipoles = self.dip
        E_grid = self.E_grid
        E_point= self.E_point 
    
        E_grid[...] = 0
        E_grid = self._spread(E_grid,dipoles)
        fE_grid = fftshift(fftn(E_grid,axes=(0,1,2),overwrite_x=True),axes=(0,1,2))

        fEs_grid = self._scale(fE_grid)
        Es_grid = ifftn(ifftshift(fEs_grid,axes=(0,1,2)),axes=(0,1,2),overwrite_x=True)
    
        E_point[...] = 0
        E_point = self._contract(E_point,Es_grid)
        E_point = self._real_space(E_point,dipoles)
        return E_point
        # }}}

    def _spread(self,E_grid,dip):# {{{
        spread_coef = self.spread_coef
        spread_idxs =  self.spread_idxs

        num_spread = spread_idxs.shape[0]
    
        Espread = spread_coef[:,:,None]*dip[:,None,:]
        Espread = Espread.reshape(num_spread,3)
    
        np.add.at(E_grid,tuple(spread_idxs.T),Espread)
        return E_grid
        # }}}
    def _scale(self,fE_grid):# {{{
        khat = self.khat
        scale_coef = self.scale_coef

        np.multiply(fE_grid,khat,out=fE_grid)
        fE_grid = fE_grid.T
        sum = scale_coef*(fE_grid[0]+fE_grid[1]+fE_grid[2]).T
        fE_grid = fE_grid.T
        #np.multiply(khat,sum[...,None],out=fE_grid)
        np.multiply(khat[...,0],sum,out=fE_grid[...,0])
        np.multiply(khat[...,1],sum,out=fE_grid[...,1])
        np.multiply(khat[...,2],sum,out=fE_grid[...,2])
        return fE_grid
    # }}}
    def _contract(self,E_point,Es_grid):# {{{
        particle_index = self.particle_index
        contract_coef = self.contract_coef
        contract_idxs =  self.contract_idxs
        np.add.at(E_point,particle_index,contract_coef[:,None]*Es_grid[*contract_idxs.T])
        return E_point
    # }}}
    def _real_space(self,E_point,dip):# {{{
        #Real
        self_perp = self.self_perp
        p1 = self.p1
        p2 = self.p2
        delta = self.delta
        para = self.para
        perp = self.perp

        if self.calc_inter_dipole:
            E_point += self.self_coef * dip
            E_point += dip*self_perp

        dip_p2 = dip[p2]
        delta_dip_p2 = np.sum(dip_p2*delta,axis = -1)
        np.add.at(E_point,p1,perp[:,None]*(dip_p2\
                                            - delta*delta_dip_p2[:,None])\
                                            + para[:,None]*delta*delta_dip_p2[:,None])
        return E_point
    # }}}
    def _spread_precalcs(self):# {{{
        num_dipoles = self.dip_pos.shape[0]
        num_spread = num_dipoles*self.offset.shape[0]
        grid_idxs = np.round(self.dip_pos/self.grid_spacing).astype(int)
        particle_grid_dist = grid_idxs*self.grid_spacing - self.dip_pos
        grid_effect_idxs = (grid_idxs[:,None,:] + self.offset[None,:,:] - 1) % self.num_grid
        spread_idxs = grid_effect_idxs.reshape(num_spread,3)

        grid_effect_dist = (particle_grid_dist[:,None,:] + self.offsetxyz[None,:,:])
        grid_effect_div_eta = np.sum(grid_effect_dist**2/self.spectral_split,axis = -1)
        spread_coef = (2*self.xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(self.spectral_split))*np.exp(-2*self.xi**2*grid_effect_div_eta)
        
        self.spread_coef = spread_coef
        self.spread_idxs = spread_idxs
        return
        # }}}
    def _contract_precalcs(self):# {{{
        num_points = self.points.shape[0]
        num_contract = num_points*self.offset.shape[0]
        grid_idxs = np.round(self.points/self.grid_spacing).astype(int)
        particle_grid_dist = grid_idxs*self.grid_spacing - self.points
        grid_effect_idxs = (grid_idxs[:,None,:] + self.offset[None,:,:] - 1) % self.num_grid
        contract_idxs = grid_effect_idxs.reshape(num_contract,3)
    
        grid_effect_dist = (particle_grid_dist[:,None,:] + self.offsetxyz[None,:,:])
        grid_effect_div_eta = np.sum(grid_effect_dist**2/self.spectral_split,axis = -1)
        contract_coef = (2*self.xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(self.spectral_split))*np.exp(-2*self.xi**2*grid_effect_div_eta)

        contract_coef = np.prod(self.grid_spacing)*contract_coef
        contract_coef = contract_coef.reshape(num_contract)
    
        particle_index = np.repeat(np.arange(num_points),len(self.offset))
        E_point = np.zeros([num_points,3]).astype(np.complex128)

        self.E_point = E_point
        self.particle_index = particle_index
        self.contract_coef = contract_coef
        self.contract_idxs = contract_idxs
        return
        # }}}
    def _real_space_precalcs(self):# {{{
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
        self.delta = r
    
        self.p1 = p1[cutoff_flags]
        self.p2 = p2[cutoff_flags]
    
        int_perp = interp1d(self.r_table,self.field_dip_1)
        int_para = interp1d(self.r_table,self.field_dip_2)
    
        self.self_perp = int_perp(0)
        self.perp = int_perp(d)
        self.para = int_para(d)
        # }}}
    def _calc_real_space_table(self):# {{{
    
        r = np.arange(1,10,.001)
        xi = self.xi

        pi = np.pi
        exp = np.exp
    
        # Polynomials multiplying the exponetials
        exppolyp = -(r+2)/(32*pi**(3/2)*xi*r)
        exppolym = -(r-2)/(32*pi**(3/2)*xi*r)
        exppoly0 = 1/(16*pi**(3/2)*xi)
        
        # Polynomials multiplying the error functions
        erfpolyp = (2*xi**2*(r+2)**2 + 1)/(64*pi*xi**2*r)
        erfpolym = (2*xi**2*(r-2)**2 + 1)/(64*pi*xi**2*r)
        erfpoly0 = -(2*xi**2*r**2 + 1)/(32*pi*xi**2*r)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r) + (4-r)/(16*pi)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        pot_charge = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        ## Potential/Dipole or Field/Charge coupling
        
        # Polynomials multiplying the exponetials
        exppolyp = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 - 4*xi**2*r**2 + (-3+8*xi**2)*r + 2*(1-8*xi**2))
        exppolym = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 + 4*xi**2*r**2 + (-3+8*xi**2)*r - 2*(1-8*xi**2))
        exppoly0 = 3*(2*r**2*xi**2+1)/(128*pi**(3/2)*xi**3*r)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 + 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
        erfpolym = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 - 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
        erfpoly0 = -3*(4*xi**4*r**4 + 4*xi**2*r**2 - 1)/(256*pi*xi**4*r**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r**2) + r/(8*pi)*(1-3/8*r)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        pot_dip = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
        ## Field/Dipole coupling: I-rr component
    
        # Polynomials multiplying the exponentials
        exppolyp = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 - \
            8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppolym = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 + \
            8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 1/(512*pi**(3/2)*xi**5*r**2)*(-4*xi**4*r**4 - 8*xi**2*(2-9*xi**2)*r**2 - 3+36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 256*xi**6*r**3 - \
            18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpolym = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 256*xi**6*r**3 - \
            18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpoly0 = 1/(1024*pi*xi**6*r**3)*(8*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 18*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -1/(4*pi*r**3) + 1/(4*pi)*(1-9*r/16+r**3/32)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        
        ## Field/Dipole coupling: rr component
        
        # Polynomials multiplying the exponentials
        exppolyp = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 - 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 - \
            4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppolym = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 + 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 + \
            4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 1/(256*pi**(3/2)*xi**5*r**2)*(-8*xi**4*r**4 - 2*xi**2*(7-36*xi**2)*r**2 + 3-36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
        erfpolym = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
        erfpoly0 = 1/(512*pi*xi**6*r**3)*(16*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 3-36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 1/(2*pi*r**3) + 1/(4*pi)*(1-9*r/8+r**3/8)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
        ## Field/Dipole Force: coefficient multiplying -(mi*mj)r and -( (mj*r)mi + (mi*r)mj - 2(mi*r)(mj*r)r )
        # Polynomials multiplying the exponentials
        exppolyp = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 + 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppolym = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 - 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 3/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 - 4*xi**2*(1-6*xi**2)*r**2 + 3-36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
        erfpolym = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
        erfpoly0 = 3/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 12*xi**4*(1-4*xi**2)*r**4 - 6*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = 3/(4*pi*r**4) - 3/(64*pi)*(3-r**2/2)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_force_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
            erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
        
        ## Field/Dipole Force from:  coefficient multiplying -(mi*r)(mj*r)r
        
        # Polynomials multiplying the exponentials
        exppolyp = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**4*r**3 + 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
        exppolym = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**4*r**3 - 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
        exppoly0 = 9/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 + 8*xi**4*r**2 - 3+36*xi**2)
        
        # Polynomials multiplying the error functions
        erfpolyp = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpolym = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
        erfpoly0 = 9/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 4*xi**4*(1-4*xi**2)*r**4 + 2*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
        
        # Regularization for overlapping particles
        if self.calc_inter_dipole:
            regpoly = -9/(4*pi*r**4) - 9/(64*pi)*(1-r**2/2)
        else:
            regpoly = 0
        
        # Combine the polynomial coefficients, exponentials, and error functions
        field_dip_force_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
    
        ## Self terms
        
        # Potential/charge
        selfo = (1-exp(-4*xi**2))/(8*pi**(3/2)*xi) + erfc(2*xi)/(4*pi)
        pot_charge = np.insert(pot_charge,0,selfo)
        
        # Potential/dipole or field/charge
        pot_dip = np.insert(pot_dip,0,0)
        
        # Field/dipole
        selfo = (-1+6*xi**2+(1-2*xi**2)*exp(-4*xi**2))/(16*pi**(3/2)*xi**3) + erfc(2*xi)/(4*pi)
        self.field_dip_1 = np.insert(field_dip_1,0,selfo)
        self.field_dip_2 = np.insert(field_dip_2,0,selfo)
        
        # Field/dipole force
        field_dip_force_1 = np.insert(field_dip_force_1,0,0)
        field_dip_force_2 = np.insert(field_dip_force_2,0,0)

        self.r_table = np.insert(r,0,0)
        return # }}}
    def _precalculations(self):# {{{
        self.rc = np.sqrt(-np.log(self.errortol))/self.xi
        if np.any(self.rc > self.box/2):
            raise Exception(f"Real space cutoff ({self.rc:.3f}) larger than half the box length.")

        self._calc_real_space_table()

        self.kcut = 2*self.xi**2*self.rc
        self.num_grid = np.ceil(1+self.box*self.kcut/np.pi).astype(int)
        self.grid_spacing = self.box/self.num_grid
        self.num_grid_gaussian = np.ceil(-2*np.log(self.errortol)/np.pi)
        self.spectral_split = self.num_grid_gaussian * (self.grid_spacing*self.xi)**2/np.pi
        self.E_grid = np.zeros(np.append(self.num_grid,3)).astype(np.complex128)

        off = int(self.num_grid_gaussian/2)
        min_off = -off
        max_off = off+1
        offset = []
        for x in range(min_off,max_off):
            for y in range(min_off,max_off):
                for z in range(min_off,max_off):
                    offset.append([x,y,z])
        self.offset = np.array(offset)[:,[2,1,0]]
        self.offsetxyz = self.offset*self.grid_spacing

        #---- Scale Precalcs ----{{{
        warnings.filterwarnings('ignore')
    
        Kx = np.arange(-np.ceil((self.num_grid[0]-1)/2),np.floor((self.num_grid[0] - 1)/2)+1) * 2*np.pi/self.box[0]
        Ky = np.arange(-np.ceil((self.num_grid[1]-1)/2),np.floor((self.num_grid[1] - 1)/2)+1) * 2*np.pi/self.box[1]
        Kz = np.arange(-np.ceil((self.num_grid[2]-1)/2),np.floor((self.num_grid[2] - 1)/2)+1) * 2*np.pi/self.box[2]
    
        k0x = np.argwhere(Kx == 0)
        k0y = np.argwhere(Ky == 0)
        k0z = np.argwhere(Kz == 0)
        k0_ind = np.array([k0x,k0y,k0z])
    
        kx,ky,kz = np.meshgrid(Kx,Ky,Kz,indexing='ij')
        k = np.concatenate([kx[:,:,:,None],ky[:,:,:,None],kz[:,:,:,None]],axis = -1)
    
        ksq = k**2
        ksqsm = np.sum(ksq,axis = -1)
        kmag = np.sqrt(ksqsm)
        khat = k/kmag[:,:,:,None]
        khat[*k0_ind] = 0
    
        etaksq  = np.sum(ksq*(1-self.spectral_split),axis = -1)
        scale_coef = 9*np.pi/(2*kmag) * besselj(1+1/2,kmag)**2 * np.exp(-etaksq/(4*self.xi**2)) / ksqsm
        scale_coef[*k0_ind] = 0
        self.scale_coef = scale_coef
        self.khat = khat
        warnings.filterwarnings('default')
        # }}}
        return 
    # }}}
    def _gen_neighbor_list(self,):#{{{
        box_length = self.box
        cutoff = self.rc
        points = self.points
        dip_pos = self.dip_pos

        if np.any(points < 0) or np.any(dip_pos < 0):
            points = np.copy(points)
            points += box_length/2

            dip_pos = np.copy(dip_pos)
            dip_pos += box_length/2
        numBoxes = (box_length/cutoff).astype(int)


        cutoff = box_length/numBoxes
        if len(points.shape) == 2:
            numFrames = 1
            numPoints = points.shape[0]
            numDips = dip_pos.shape[0]
            dims = points.shape[1]
        elif len(points.shape) == 3:
            numFrames = points.shape[0]
            numPoints = points.shape[1]
            numDips = dip_pos.shape[0]
            dims = points.shape[2]

        flag = numBoxes <= 3
        if np.all(flag):
            point_idx = np.arange(numPoints)
            dip_idx = np.arange(numDips)
            p1 = np.repeat(point_idx,numDips)
            p2 = np.tile(dip_idx,numPoints)
            flag = p1 != p2
            return p1[flag],p2[flag]
        elif np.any(flag):
            points = points[:,~flag]
            dip_pos = dip_pos[:,~flag]
            numBoxes = numBoxes[~flag]
            cutoff = cutoff[~flag]
            dims = np.count_nonzero(~flag)

        point_indices = (points/cutoff).astype(int)
        if not np.iterable(numBoxes):
            numBoxes = np.array([numBoxes]*dims)
        boxes = {}
        count = np.zeros(numBoxes)
        for i,idx in enumerate(point_indices):
            index = tuple(idx.tolist())
            if index not in boxes:
                boxes[index] = list()
            boxes[index].append(int(i))
            count[index] += 1
        P1,P2 = [],[]
        offset = np.array([np.arange(-1,2) for i in range(dims)]).T

        dip_pos_indices = (dip_pos/cutoff).astype(int)
        for p1,idx in enumerate(dip_pos_indices):
            offsets = (idx+offset)%numBoxes
            for off in itertools.product(*offsets.T):
                if count[off] == 0:
                    continue
                p2s = boxes[off]
                P1 += [p1]*len(p2s)
                P2 += (p2s)
        P1 = np.array(P1)
        P2 = np.array(P2)
        flags = P1 != P2
        P1 = P1[flags]
        P2 = P2[flags]
        return P2,P1
        #}}}
# }}}
