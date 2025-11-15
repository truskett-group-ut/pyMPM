import numpy as np
def drude_dielectric(omega,gamma,omega_p,eps_inf):# {{{
    '''
    Calculate the drude dielectric function.

    **Parameters**
        *omega*
            A 1D array of wavenumbers
        *gamma*
            A scalar or 1D array of damping coefficient.
            If an array, size must match that of *omega_p* and *eps_inf*.
        *omega_p*
            A scalar or 1D array of plasma frequency.
            If an array, size must match that of *gamma* and *eps_inf*.
        *eps_inf*
            A scalar or 1D array of infinite frequency dielectric constant.
            If an array, size must match that of *omega_p* and *gamma*.

    **Returns**
        *eps*
            A 2D array (num_wavenumbers, num_gammas) of drude dielectric functions.
            Axes of size one are removed.
    '''
    if not np.iterable(omega):
        omega = np.asarray([omega])
    if len(omega.shape) != 1:
        raise Exception("omega must be a scalar or a 1-D array of wavenumbers")
    num_wavenumbers = len(omega)

    if not np.iterable(gamma):
        gamma = np.asarray([gamma])
    if len(gamma.shape) != 1:
        raise Exception("gamma must be a scalar or a 1-D array of damping coefficients")

    if not np.iterable(omega_p):
        omega_p = np.asarray([omega_p])
    if len(omega_p.shape) != 1:
        raise Exception("omega_p must be a scalar or a 1-D array of plasma frequencies")

    if not np.iterable(eps_inf):
        eps_inf = np.asarray([eps_inf])
    if len(eps_inf.shape) != 1:
        raise Exception("eps_inf must be a scalar or a 1-D array of high frequency permitivities")

    if np.any(len(eps_inf) != np.array([len(gamma),len(omega_p)])):
        raise Exception("eps_inf, gamma, and omega_p must all have the same number of values applied")
    num_particles = len(eps_inf)

    eps_p = np.zeros([num_wavenumbers,num_particles],dtype = np.complex128)
    eps_p[...] = eps_inf[None,:] - omega_p[None,:]**2/(omega[:,None]**2+1j*omega[:,None]*gamma[None,:])
    return np.squeeze(eps_p)
