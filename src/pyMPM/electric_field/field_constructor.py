import numpy as np
import itertools

class Electric_Field():#{{{
    def __new__(self,*args,method=None,**kwargs):# {{{
        if method is None:
            if "errortol" in kwargs or "xi" in kwargs:
                from .ewald import Dipole_Field
                return Dipole_Field(*args,**kwargs)
            elif "cutoff" in kwargs:
                from .direct import Dipole_Field
                return Dipole_Field(*args,**kwargs)
        elif method == "ewald":
            from .ewald import Dipole_Field
            return Dipole_Field(*args,**kwargs)
        elif method == "direct":
            from .direct import Dipole_Field
            return Dipole_Field(*args,**kwargs)
        else:
            raise Exception("Couldn't determine method")

        # }}}
# }}}
