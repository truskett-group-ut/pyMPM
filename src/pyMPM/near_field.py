import numpy as np
from .electric_field import Electric_Field
from .neighbor_list import MultiNeighborList

class Near_Field:
    '''
    Calculate the electric field at *points* for the given *dipoles* at *dip_pos*.

    **Parameters**
        *box*
            A 1D array of length three specifying *Lx*, *Ly*, and *Lz*.
        *E0*
            The incident field. 1D array of length three.
        *dip*
            A 2D array (num_dipoles,3) specifying the dipoles contributing to the electric field.
            Can also be set with **set_dipoles** method, but must be set before calling **calculate**.
        *dip_pos*
            A 2D array (num_dipoles,3) specifying the position of the dipoles contributing to the electric field.
            Can also be set with **set_dipoles** method, but must be set before calling **calculate**.
        *field*
            A 2D (num_field_points, 3) or 3D (num_X,num_Y,3) specifying the field points where the electric field should be calculated.
            Can instead supply *field_min*, *field_max*, and *num_field* to define the field.
            Can also be set with **set_dipoles** method, but must be set before calling **calculate**.
        *xi*
            Ewald parameter. Defaults to 0.5
        *errortol*
            The error tolerance. Defaults to 0.001.
    '''
    def __init__(self,box,E0,dip=None,dip_pos=None,field=None,xi=0.5,errortol=1e-3,field_min=None,field_max=None,num_field=None):# {{{
        self.box = box
        self.dip = dip
        self.dip_pos = dip_pos
        self.E0 = E0
        self.xi = xi
        self.errortol = errortol

        if field is None and field_min is not None:
            ps = np.linspace(field_min,field_max,num_field)
            field = np.array(np.meshgrid(ps,ps)).T
        self.set_field_points(field)

        self.field_calculator = Electric_Field(self.box,self.xi,self.errortol)

        self.field_update = True
        self.dip_pos_update = True
        self.dip_update = True
        return# }}}
    
    def set_dipoles(self,dip):# {{{
        '''
        Used to set or change the dipoles.

        **Parameters**
            *dip*
                Dipoles contributing to the electric field.
        '''
        self.dip = dip
        self.dip_update = True# }}}
    def set_dipole_positions(self,dip_pos):# {{{
        '''
        Used to set or change the dipole positions.

        **Parameters**
            *dip_pos*
                The position of the dipoles contributing to the electric field.
        '''
        self.dip_pos = dip_pos
        self.dip_pos_update = True# }}}
    def set_field_points(self,field):# {{{
        '''
        Used to set or change the field points.

        **Parameters**
            *field*
                The field points where the electric field should be calculated.
        '''
        if field is None:
            self.field = field
        elif len(field.shape) == 3:
            self.idx = np.array([[[j,i] for i in range(field.shape[0])] for j in range(field.shape[1])])
            self.whole_map = True
            self.whole_field = field
            self.field = self.whole_field.reshape(np.prod(self.whole_field.shape[:-1]),2)
        else:
            self.field = field
            self.whole_map = False

        self.field_update = True
        # }}}
    def _setup_field(self):# {{{
        if self.field is None:
            raise Exception("Field must be set before calculation")
        if self.field_update or self.dip_pos_update:
            if self.whole_map:
                num_whole_field = np.prod(self.whole_field.shape[:2])
                field = self.whole_field.reshape(num_whole_field,2)
                idx = self.idx.reshape(num_whole_field,2)
                field = np.append(field,np.zeros(num_whole_field)[:,None],axis=1)
                nl = MultiNeighborList(0.95,self.box)
                p1,p2 = nl(field,self.dip_pos)
                d = np.linalg.norm(field[p1]-self.dip_pos[p2],axis=1)
                flags = np.ones(field.shape[0],dtype = bool)
                flags[p1[d<0.9]] = False
                self.field = field[flags]
                self.idx = idx[flags]

            self.field_calculator.set_points(self.field)
            if self.dip_pos_update:
                self.field_calculator.set_dip_pos(self.dip_pos)

        if self.dip_update:
            self.field_calculator.set_dipoles(self.dip)

        self.field_update = False
        self.dip_pos_update = False
        self.dip_update = False
        return# }}}
    def calculate(self,):# {{{
        '''
        Calculates the electric field intensity at *field* points.

        **Returns**
            *field*
                The points where the field was calculated at.
            *E*
                The electric field intensity at each field point.
        '''
        self._setup_field()
            
        E = -self.field_calculator.calculate() + self.E0[None,:]

        E = np.sum(np.abs(E)**2,axis = 1)
        if self.whole_map:
            Es = np.nan + np.zeros(self.whole_field.shape[:-1])
            Es[self.idx[:,1],self.idx[:,0]] = E[...]
            return self.whole_field,Es
        else:
            return E
        # }}}

if __name__ == "__main__":
    box = np.array([30,30,30])
    dip_pos = np.array(
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

    C = np.average(np.load("dip.npy")[0,:,:,1,1],axis=1)
    ext = np.imag(C)*np.arange(0.01,0.81,0.01)
    arg = np.argmax(ext)
    dip = np.load("dip.npy")[0,arg,:,1]

    xi = 0.5
    errortol = 1e-3
    E0 = np.array([0,1,0])

    nf = Near_Field(box,E0,dip=dip,dip_pos=dip_pos,field_min=-5.6,field_max=5.6,num_field=500,xi=0.5,errortol=1e-3)
    pts,E = nf.calculate()

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib as mpl
    import matplotlib.patches as patch
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    cmap = ax.pcolormesh(*pts.T,np.log(E),cmap=plt.cm.inferno)
    plt.colorbar(cmap,ax=ax)
    for p in dip_pos:
        ps = patch.Circle(p,1,color='k')
        ax.add_patch(ps)
    ax.quiver(dip_pos[:,0],dip_pos[:,1],np.imag(dip[:,0]),np.imag(dip[:,1]),color="w",pivot="mid")
    plt.show()
