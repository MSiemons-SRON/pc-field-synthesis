from pathlib import Path
from parameters import load_field_data, get_pupil_params
from synthesis import get_transmission, get_flux, integrate2d, get_poynting_vector
import numpy as np
import h5py
from scipy.interpolate import interpn

if __name__ == "__main__":
    save_directory = Path("Figures and Data/Data/")

    # %% Transmission field synthesis
    for mesh_accuracy in [2,3]:
        directory = Path(rf"E:\Marijn\PlaneWave_paperproject\PlanewaveDecomposition\Results\MeshAccuracy{mesh_accuracy}")

        n_rings = [30, 15, 10, 6, 5, 3]
        transmissions = np.zeros((len(n_rings), 301))

        for i, n_ring in enumerate(n_rings):
            params = {"filename": directory / f"AngledPlanewaveSweep_{n_ring}rings.h5",
                      "fnumber": 5.7,
                      "pupilshape": "gaussian",
                      "distance_from_focus": 1000}
            params = load_field_data(params)
            params = get_pupil_params(params)
            transmissions[i,:] = get_transmission(params)

        # %% load focused beam result
        focusedbeam_file = Path(
            fr"E:\Marijn\PlaneWave_paperproject\Gaussianbeam\GausianBeam_25deg_Lc1000nm_width500nm_film415nm_50000fsPulse_mesh{mesh_accuracy}")

        with h5py.File(focusedbeam_file, 'r') as f:
            transmission_focusedbeam = np.squeeze(f["T"][()])
            wavelengths_focusedbeam = np.squeeze(f["wavelength"][()]) * 1e3

        with h5py.File(directory / "AngledPlanewaveSweep_30rings.h5", 'r') as f:
            transmission_planewave = np.squeeze(f["t_store"][0, :])

        rms_error = np.sqrt(np.sum((transmissions.T - transmission_focusedbeam[:, None]) ** 2, axis=0) / len(
                transmission_focusedbeam))

        with h5py.File(save_directory/f"field_synthesis_transmission_mesh{mesh_accuracy}.h5", "w") as f:
            f.create_dataset(name="transmissions", data=transmissions)
            f.create_dataset(name="n_rings", data=n_rings)
            f.create_dataset(name="wavelengths", data=params["wavelengths"])
            f.create_dataset(name="transmission_planewave", data=transmission_planewave)
            f.create_dataset(name="transmission_focusedbeam", data=transmission_focusedbeam)
            f.create_dataset(name="rms_error", data=rms_error)

    # %% Figure 2 data
    with h5py.File(directory / "AngledPlanewaveSweep_30rings.h5", mode='r') as f:
        transmission_planewaves = np.squeeze(f["t_store"])
        theta = np.squeeze(f["theta"][()])
        phi = np.squeeze(f["phi"][()])
    p_pol_index = phi == 0
    s_pol_index = phi == 90

    transmission_ppol = transmission_planewaves[p_pol_index]
    transmission_spol = transmission_planewaves[s_pol_index]
    theta_ppol = theta[p_pol_index]
    theta_spol = theta[s_pol_index]

    with h5py.File(save_directory / f"transmission_s_and_p_pol.h5", mode="w") as f:
        f.create_dataset(name="transmission_ppol", data=transmission_ppol)
        f.create_dataset(name="transmission_spol", data=transmission_spol)
        f.create_dataset(name="theta_ppol", data=theta_ppol)
        f.create_dataset(name="theta_spol", data=theta_spol)

    # %% Figure 4
    # Load Field synthesis file and compute PSF
    directory = Path(r"E:\Marijn\PlaneWave_paperproject\PlanewaveDecomposition\Results\MeshAccuracy3")

    params = {"filename": directory / "AngledPlanewaveSweep_30rings.h5",
              "fnumber": 5.7,
              "pupilshape": "gaussian",
              "distance_from_focus": 1000}
    params = load_field_data(params)
    params = get_pupil_params(params)

    # Unpack e and h fields from parameters
    e_mon = params['e_mon']
    h_mon = params['h_mon']
    e_src = params['e_src']
    h_src = params['h_src']

    # compute PSF for wavelengths 1800 and 1687 nm and src PSF
    wavelength_ind_1 = 0
    wavelength_ind_2 = 100

    flux_src, psf_src = get_flux(e_src, h_src, params, wavelength_ind=150, psf=True)
    flux_1, psf_1 = get_flux(e_mon[:, wavelength_ind_1], h_mon[:, wavelength_ind_1], params, wavelength_ind_1, psf=True)
    flux_2, psf_2 = get_flux(e_mon[:, wavelength_ind_2], h_mon[:, wavelength_ind_2], params, wavelength_ind_2, psf=True)

    # Scale PSF such that integrating them gives transmission value
    psf_norm_1 = -0.5 * psf_1 / params["source_power"][wavelength_ind_1] / flux_src
    psf_norm_2 = -0.5 * psf_2 / params["source_power"][wavelength_ind_2] / flux_src


    # Load focused beam file
    focusedbeam_file = Path(
        r"E:\Marijn\PlaneWave_paperproject\Gaussianbeam\GausianBeam_25deg_Lc1000nm_width500nm_film415nm_50000fsPulse_mesh3_nosym_notextended.mat")

    with h5py.File(focusedbeam_file, 'r') as f:
        transmission_focusedbeam = np.squeeze(f["T"][()])
        wavelengths_focusedbeam = np.squeeze(f["wavelength"][()]) * 1e3
        e_x = f["Ex"]["real"][()] + 1j * f["Ex"]["imag"][()]
        e_y = f["Ey"]["real"][()] + 1j * f["Ey"]["imag"][()]
        h_x = f["Hx"]["real"][()] + 1j * f["Hx"]["imag"][()]
        h_y = f["Hy"]["real"][()] + 1j * f["Hy"]["imag"][()]
        x_gaussian_beam = np.squeeze(f["xvector"][()])

    # Compute poynting vector based on e and h fields
    e_field_gaussian = np.zeros(shape=(301, 3, *e_x.shape[-3:]), dtype="complex64")
    e_field_gaussian[:, 0, :, :] = e_x
    e_field_gaussian[:, 1, :, :] = e_y
    h_field_gaussian = np.zeros(shape=(301, 3, *e_x.shape[-3:]), dtype="complex64")
    h_field_gaussian[:, 0, :, :] = h_x
    h_field_gaussian[:, 1, :, :] = h_y
    gaussian_psf = np.squeeze(np.cross(e_field_gaussian, np.conj(h_field_gaussian), axis=1)[:, 2])

    # Interpolate focused beam PSF onto field synthesis mesh and scale similar to field synthesis
    xi, yi = np.meshgrid(params["xvector_field"], params["xvector_field"])
    pointi = np.array([xi.flatten(), yi.flatten()])
    x = (x_gaussian_beam, x_gaussian_beam)
    gaussian_psf_intp_1 = interpn(x, -gaussian_psf[wavelength_ind_1].real, pointi.T).reshape(
        len(params["xvector_field"]), len(params["xvector_field"]))
    gaussian_psf_intp_2 = interpn(x, -gaussian_psf[wavelength_ind_2].real, pointi.T).reshape(
        len(params["xvector_field"]), len(params["xvector_field"]))
    gaussian_psf_intp_norm_1 = (gaussian_psf_intp_1
                                / integrate2d(params["xvector_field"], params["xvector_field"], gaussian_psf_intp_1)
                                * transmission_focusedbeam[wavelength_ind_1])
    gaussian_psf_intp_norm_2 = (gaussian_psf_intp_2
                                / integrate2d(params["xvector_field"], params["xvector_field"], gaussian_psf_intp_2)
                                * transmission_focusedbeam[wavelength_ind_2])

    # Save
    with h5py.File(save_directory / f"psf_comparison.h5", mode="w") as f:
        f.create_dataset(name="psf_focusedbeam_1", data=gaussian_psf_intp_norm_1)
        f.create_dataset(name="psf_focusedbeam_2", data=gaussian_psf_intp_norm_2)
        f.create_dataset(name="x_vector", data=params["xvector_field"])
        f.create_dataset(name="psf_fieldsynthesis_1", data=psf_norm_1)
        f.create_dataset(name="psf_fieldsynthesis_2", data=psf_norm_1)

    # %% Figure 5 Pupil PSFs
    directory = Path(r"E:\Marijn\PlaneWave_paperproject\PlanewaveDecomposition\Results\MeshAccuracy3")

    pupil_shapes = ["gaussian", "airy", "annulus"]

    transmissions_pupils = np.zeros((3, 301))
    psh_pupils = []

    for i, shape in enumerate(pupil_shapes):
        params = {"filename": directory / "AngledPlanewaveSweep_30rings.h5",
                  "fnumber": 5.7,
                  "pupilshape": shape,
                  "distance_from_focus": 1000}
        params = load_field_data(params)
        params = get_pupil_params(params)

        transmissions_pupils[i,:] = get_transmission(params)

        jlambda = 130
        psh_pupils.append(get_poynting_vector(params, wavelength_ind=jlambda))

    psh_pupils = np.array(psh_pupils)

    # Save
    with h5py.File(save_directory / f"psf_pupil_comparison.h5", mode="w") as f:
        f.create_dataset(name="psh_pupils", data=psh_pupils)
        f.create_dataset(name="transmissions_pupils", data=transmissions_pupils)
        f.create_dataset(name="wavelengths", data=params["wavelengths"])
        f.create_dataset(name="x_vector", data=params["xvector_field"])





