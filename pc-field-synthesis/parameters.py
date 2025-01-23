import numpy as np
from aberration import get_zernike_mode
import h5py

def get_pupil_params(params):
    """
    Computes pupil function for different input parameters

    Parameters
    ----------
    params : dict
        Dict containing pupilshape, fnumber, distance_from_focus, center_wavelength, lattice_constant.

    Returns
    -------
    pupil : 2D np.array
        Pupil function for the devined pupil shape and f-number.
    params : dict
        dictionary containing parameters

    """

    pupilshape = params.get("pupilshape")
    fnumber = params.get("fnumber")
    theta_deg = params.get("theta")
    theta_rad = theta_deg / 180 * np.pi
    phi_deg = params.get("phi")
    phi_rad = phi_deg / 180 * np.pi
    max_theta = np.max(theta_deg) / 180 * np.pi
    angles_x_rad = theta_rad * np.cos(phi_rad)
    angles_y_rad = theta_rad * np.sin(phi_rad)
    distance_from_focus = params.get("distance_from_focus")
    center_wavelength = params.get("center_wavelength")
    max_focus_angle_rad = np.arctan(1 / (2 * fnumber))
    lattice_constant = params['lattice_constant']

    if pupilshape == "gaussian":
        # aperture_mask = theta_rad <= max_focus_angle_rad
        apodization = np.exp(-np.sin(theta_rad)**2 / (np.sin(0.5*max_focus_angle_rad))**2)
        apodization = apodization / np.sum(apodization)

    elif pupilshape == "airy":
        aperture_mask = theta_rad <= max_focus_angle_rad
        apodization = aperture_mask / np.sum(aperture_mask)

    if params.get("order") and params.get("aberration_level"):
        aberration = params.get("aberration_level") * get_zernike_mode(params.get("order"), theta_rad, phi_rad)
    else:
        aberration = 0
    phase = (np.sqrt(1 - np.sin(theta_rad)**2) - 1) * distance_from_focus * np.pi * 2 / center_wavelength
    pupil = apodization * np.exp(-1j*phase) * np.exp(2*np.pi*1j*aberration)

    # Compute additional params
    NA = np.sin(max_focus_angle_rad)
    airy_2nd_radius = 7.0 / np.pi / 2 * center_wavelength / NA
    if pupilshape == 'gaussian':
        psf_size = center_wavelength / np.pi / max_focus_angle_rad
    else:
        psf_size = center_wavelength / (2 * NA)

    Npc = int(np.ceil(2*airy_2nd_radius / lattice_constant))
    x_image = np.linspace(-(Npc-0.5) * lattice_constant, (Npc-0.5) * lattice_constant, 2*Npc)
    xx_image, yy_image = np.meshgrid(x_image, x_image)
    xvector_field = np.linspace(-Npc * lattice_constant, Npc * lattice_constant, 2*Npc*(len(params.get("xvector"))-1)+1)*1e-9

    params['angles_x_rad'] = angles_x_rad
    params['angles_y_rad'] = angles_y_rad
    params['xx_image'] = xx_image
    params['yy_image'] = yy_image
    params['xvector_field'] = xvector_field
    params['NA'] = NA
    params['psf_size'] = psf_size
    params["psf_simulation_size"] = airy_2nd_radius
    params['Npc'] = Npc
    params['pupil'] = pupil

    return params


def load_field_data(params):
    """
    Load EM field data

    Parameters
    ----------
    params

    Returns
    -------

    """
    filename = params.get("filename")

    with h5py.File(filename) as f:
        e_mon = f["e_mon"][()]
        h_mon = f["h_mon"][()]

        e_src = f["e_src"][()]
        h_src = f["h_src"][()]

        xvector = np.squeeze(f["xvector"][()])
        yvector = np.squeeze(f["yvector"][()])
        theta = np.squeeze(f["theta"][()])
        phi = np.squeeze(f["phi"][()])
        source_power = np.squeeze(f["src_power"][()])
        wavelengths = np.squeeze(f["wavelengths"]) * 1000
        center_wavelength = np.squeeze(f['src_wavelength'][()])
        planewave_transmission = np.squeeze(f["t_store"])

        max_sim_angle_deg = np.squeeze(f['simulation_parameters/sweepparameters/max_angle_deg'][:])
        n_rings = np.squeeze(f['simulation_parameters/sweepparameters/number_of_rings'][()])
        lattice_constant = np.squeeze(f['simulation_parameters/photoniccrystal/lattice_constant']) * 1e9
        pc_shape_char = f['simulation_parameters/photoniccrystal/shape'][:]
        pc_shape = "".join([chr(asci[0]) for asci in pc_shape_char])
        pc_size = np.squeeze(f['simulation_parameters/photoniccrystal/width'][()]) * 1e9

    params['pc_shape'] = pc_shape
    params['pc_size'] = pc_size
    params["wavelengths"] = wavelengths
    params["source_power"] = source_power
    params["xvector"] = xvector
    params["yvector"] = yvector
    params["theta"] = theta
    params["phi"] = phi
    params["max_sim_angle_deg"] = max_sim_angle_deg
    params["n_rings"] = int(n_rings)
    params['lattice_constant'] = lattice_constant
    params['center_wavelength'] = center_wavelength
    params['center_wavelength_index'] = int(np.squeeze(np.argwhere(wavelengths == center_wavelength)))
    params['planewave_transmission'] = planewave_transmission
    params['e_mon'] = e_mon
    params['h_mon'] = h_mon
    params['e_src'] = e_src
    params['h_src'] = h_src

    return params
