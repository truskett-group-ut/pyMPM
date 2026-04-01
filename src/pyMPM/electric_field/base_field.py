import numpy as np
import time
import itertools

class Base_Electric_Field():#{{{
    def __init__(self,box,eps_p=None,points = None,dip = None,dip_pos=None,k=None,R=None,debug=False):# {{{
        self.box = box

        self.dip = dip
        self.dip_pos = dip_pos

        if isinstance(debug,bool):
            if debug:
                self.debug = []
            else:
                self.debug = []
        elif isinstance(debug,list):
            self.debug = debug
        else:
            self.debug = [debug]

        self.R = R
        if R is None:
            self.has_new_R = False
        else:
            self.has_new_R = True

        if k is None:
            self.has_new_k = False
        else:
            self.k = k
            self.has_new_k = True

        if eps_p is None:
            self.calc_inter_dipole = False
            self.has_new_eps_p = False
        else:
            self.eps_p = eps_p
            self.calc_inter_dipole = True
            self.has_new_eps_p = True
        # }}}
    def set_dipoles(self,dip):# {{{
        self.dip = dip
        # }}}
    def set_dip_pos(self,dip_pos):# {{{
        self.dip_pos = dip_pos
        self.has_new_dip_pos = True
    # }}}
    def set_points(self,points):# {{{
        self.points = points
        self.has_new_points = True
    # }}}
    def set_eps_p(self,eps_p):# {{{
        self.has_new_eps_p = True
        self.calc_inter_dipole = True
        self.eps_p = eps_p
        # }}}
    def set_k(self,k):# {{{
        self.k = k
        self.has_new_k = True
    # }}}
    def set_R(self,R):# {{{
        self.R = R
        self.has_new_R = True
    # }}}
    def calculate(self):# {{{
        return 
    # }}}
    def _electric_field(self):#{{{
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
            box_length = box_length[~flag]
            dims = np.count_nonzero(~flag)

        cutoff = box_length/numBoxes
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
        if self.calc_inter_dipole:
            flags = P1 != P2
            P1 = P1[flags]
            P2 = P2[flags]
        return P2,P1
        #}}}
    def _debug(self,category,*args):# {{{
        if category.lower() not in self.debug:
            return
        print(f"{category.upper()}:",*args)
        # }}}
# }}}
