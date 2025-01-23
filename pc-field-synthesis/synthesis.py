import numpy as np
import numba
from multiprocessing import Pool

def get_poynting_vector(params, wavelength_ind, compute_src=False):
    """
    Standalone function to compute the poynting vector for a given parameter set.

    Parameters
    ----------
    e_pwd
    h_pwd
    params
    wavelength_ind

    Returns
    -------

    """
    center_wavelength = params["center_wavelength"]
    yy_image = params["yy_image"]
    yvector = params["yvector"]
    xvector = params["xvector"]
    xvector_field = params["xvector_field"]
    xx_image = params["xx_image"]
    angles_x_rad = params["angles_x_rad"]
    angles_y_rad = params["angles_y_rad"]
    pupil = params["pupil"]
    wavelength = params["wavelengths"][wavelength_ind]

    if not compute_src:
        e_pwd, h_pwd = params['e_mon'], params['h_mon']
    else:
        e_pwd, h_pwd = params['e_src'], params['h_src']

    # Pre allocate field data
    e_psf = np.zeros(shape=(3, len(xvector_field), len(xvector_field)), dtype='complex64')
    h_psf = np.zeros(shape=(3, len(xvector_field), len(xvector_field)), dtype='complex64')

    # Pre compute several variables
    kx = 2 * np.pi / wavelength * np.sin(angles_x_rad * wavelength / center_wavelength)
    ky = 2 * np.pi / wavelength * np.sin(angles_y_rad * wavelength / center_wavelength)
    e_pwd = e_pwd * pupil[:, None, None, None]
    h_pwd = h_pwd * pupil[:, None, None, None]

    # Loop over PC unit cells, compute plane-wave integral and add to field
    for ix in range(xx_image.shape[0]):
        for jy in range(yy_image.shape[1]):
            x = xx_image[ix, jy]
            y = yy_image[ix, jy]

            phase_factor = np.exp(-1j*(kx*x + ky*y))

            # indices of E and H-field
            if (ix == 0):
                indx = [ix*(len(xvector)-1), ix*(len(xvector)-1)+len(xvector)]
                start_indx = 0
            else:
                indx = [1+ix*(len(xvector)-1), ix*(len(xvector)-1)+len(xvector)]
                start_indx = 1

            if (jy == 0):
                indy = [jy*(len(yvector)-1), jy*(len(yvector)-1)+len(yvector)]
                start_indy = 0
            else:
                indy = [1+jy*(len(yvector)-1), jy*(len(yvector)-1)+len(yvector)]
                start_indy = 1

            e_pupil = e_pwd[:, :2, start_indx:, start_indy:] * phase_factor[:, None, None, None]
            h_pupil = h_pwd[:, :2, start_indx:, start_indy:] * phase_factor[:, None, None, None]

            e_psf[:2, indx[0]:indx[1], indy[0]:indy[1]] = np.sum(e_pupil, axis=0)
            h_psf[:2, indx[0]:indx[1], indy[0]:indy[1]] = np.sum(h_pupil, axis=0)

    s_z = np.cross(e_psf, np.conj(h_psf), axis=0)[2]

    return s_z


@numba.jit(nopython=True)
def integrate2d(x, y, z):
    """
    Integrates z = f(x,y) with trapizoidal weight as used in Lumerical FDTD.
    Can handled uniform and non-uniform sampling.

    Parameters
    ----------
    x : 1D np array
        position vector, but be increasing.
    y : 1D np array
        position vector, but be increasing.
    z : 2D np array
        Values to be integrated

    Returns
    -------
    s : float
        Integration value


    """
    s = 0
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            dx = x[i] - x[i-1]
            dy = y[j] - y[j-1]
            s += 1 / 4 * dx * dy * (z[j, i] + z[j, i-1] + z[j-1, i] + z[j-1, i-1])

    return s


def get_transmission(params, process_pool=8):
    """
    Computes the transmission for a parameter set.
    process_pool sets the number of parallel processors to compute the transmission.
    Computation might more than 30min.

    Parameters
    ----------
    params
    process_pool

    Returns
    -------

    """
    e_src, h_src = params["e_src"], params["h_src"]
    flux_src = get_flux(e_src, h_src, params, params["center_wavelength_index"])
    iterable = [(params["e_mon"][:, jlambda], params["h_mon"][:, jlambda], params, jlambda) for jlambda in range(len(params.get('wavelengths')))]

    with Pool(processes=process_pool) as p:
        flux_mon = p.starmap(get_flux, iterable)

    # Normalize with source flux and scale with wavelength dependent source power
    transmission = np.array(flux_mon) / flux_src * params["source_power"] / np.max(params["source_power"])

    return transmission


def get_flux(e_pwd, h_pwd, params, wavelength_ind, psf=False):
    """
    Computes the EM-flux through the monitor.

    Parameters
    ----------
    e_pwd
    h_pwd
    params
    wavelength_ind
    psf

    Returns
    -------

    """

    # Functions inside for parallel processing #

    @numba.jit(nopython=True)
    def integrate2d(x, y, z):
        """
        Integrates z = f(x,y) with trapizoidal weight as used in Lumerical FDTD.
        Can handled uniform and non-uniform sampling.

        Parameters
        ----------
        x : 1D np array
            position vector, but be increasing.
        y : 1D np array
            position vector, but be increasing.
        z : 2D np array
            Values to be integrated

        Returns
        -------
        s : float
            Integration value


        """
        s = 0
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                dx = x[i] - x[i-1]
                dy = y[j] - y[j-1]
                s += 1 / 4 * dx * dy * (z[j, i] + z[j, i-1] + z[j-1, i] + z[j-1, i-1])

        return s

    def get_poynting_vector(e_pwd, h_pwd, params, wavelength_ind):

        center_wavelength = params["center_wavelength"]
        yy_image = params["yy_image"]
        yvector = params["yvector"]
        xvector = params["xvector"]
        xvector_field = params["xvector_field"]
        xx_image = params["xx_image"]
        angles_x_rad = params["angles_x_rad"]
        angles_y_rad = params["angles_y_rad"]
        pupil = params["pupil"]
        n_rings = params["n_rings"]
        wavelength = params.get('wavelengths')[wavelength_ind]

        # Pre allocate field data
        e_psf = np.zeros((3, len(xvector_field), len(xvector_field)), dtype='complex64')
        h_psf = np.zeros((3, len(xvector_field), len(xvector_field)), dtype='complex64')

        # Pre compute several variables
        kx = 2 * np.pi / wavelength * np.sin(angles_x_rad * wavelength / center_wavelength)
        ky = 2 * np.pi / wavelength * np.sin(angles_y_rad * wavelength / center_wavelength)
        e_pwd = e_pwd * pupil[:, None, None, None]
        h_pwd = h_pwd * pupil[:, None, None, None]

        # Loop over PC unit cells, compute plane-wave integral and add to field
        for ix in range(xx_image.shape[0]):
            for jy in range(yy_image.shape[1]):
                x = xx_image[ix, jy]
                y = yy_image[ix, jy]

                phasefactor = np.exp(-1j*(kx*x + ky*y))

                # indices to stitch  E and H-field
                if ix == 0:
                    indx = [ix*(len(xvector)-1), ix*(len(xvector)-1)+len(xvector)]
                    start_indx = 0
                else:
                    indx = [1+ix*(len(xvector)-1), ix*(len(xvector)-1)+len(xvector)]
                    start_indx = 1

                if jy == 0:
                    indy = [jy*(len(yvector)-1), jy*(len(yvector)-1)+len(yvector)]
                    start_indy = 0
                else:
                    indy = [1+jy*(len(yvector)-1), jy*(len(yvector)-1)+len(yvector)]
                    start_indy = 1

                # compute pupil fields
                e_pupil = e_pwd[:, :2, start_indx:, start_indy:] * phasefactor[:, None, None, None]
                h_pupil = h_pwd[:, :2, start_indx:, start_indy:] * phasefactor[:, None, None, None]

                # Integrate pupil
                e_psf[:2, indx[0]:indx[1], indy[0]:indy[1]] = np.sum(e_pupil, axis=0)
                h_psf[:2, indx[0]:indx[1], indy[0]:indy[1]] = np.sum(h_pupil, axis=0)

        s_z = np.cross(e_psf, np.conj(h_psf), axis=0)[2]

        return s_z

    # Get data out
    source_power = params.get('source_power')
    xvector_field = params.get('xvector_field')

    # Compute poynting vector
    s_z = get_poynting_vector(e_pwd, h_pwd, params, wavelength_ind)

    # Compute transmission
    flux = - 1 / 2 * integrate2d(xvector_field, xvector_field, s_z.real) / source_power[wavelength_ind]

    if psf:
        return flux, s_z.real
    return flux
