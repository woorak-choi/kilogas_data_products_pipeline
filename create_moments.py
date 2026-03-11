from astropy.io import fits
import numpy as np
from glob import glob
import os
from astropy.cosmology import FlatLambdaCDM


def innersquare(cube):
    """
    Get the central square (in spatial directions) of the spectral cube. Can
    also be used on a 2D array. If the central part is empty, the original
    array is returned.

    Parameters
    ----------
    cube : 2D or 3D numpy array
        3D array input cube or 2D image

    Returns
    -------
    cube : 2D or 3D array
        2D or 3D array of the inner 1/8 of the cube in the spatial directions
    """

    if len(cube.shape) == 3:
        start_x = int(cube.shape[1] / 2 - cube.shape[1] / 8)
        stop_x = int(cube.shape[1] / 2 + cube.shape[1] / 8)
        start_y = int(cube.shape[2] / 2 - cube.shape[1] / 8)
        stop_y = int(cube.shape[2] / 2 + cube.shape[1] / 8)
        inner_square = cube[:, start_x:stop_x, start_y:stop_y]

        # if "inner_square" is empty, return the cube instead.
        if (inner_square == inner_square).any():
            return inner_square
        else:
            return cube

    elif len(cube.shape) == 2:
        start_x = int(cube.shape[0] / 2 - 20)
        stop_x = int(cube.shape[0] / 2 + 20)
        start_y = int(cube.shape[1] / 2 - 20)
        stop_y = int(cube.shape[1] / 2 + 20)
        inner_square = cube[start_x:stop_x, start_y:stop_y]
        if (inner_square == inner_square).any():
            return inner_square
        else:
            return cube

    else:
        raise AttributeError("Please provide a 2D or 3D array.")


def new_header(header):
    """
    Change the 3D header to the corresponding 2D one, assuming the velocity
    axis has been collapsed.

    Parameters
    ----------
    header : FITS header
        The original FITS header containing 3D information.

    Returns
    -------
    header : FITS header
        The same header but with 3D information removed.

    """

    header = header.copy()

    header["NAXIS"] = 3

    try:
        header["WCSAXES"] = 2
    except:
        pass

    header.pop("CTYPE3")
    header.pop("CRVAL3")
    header.pop("CDELT3")
    header.pop("CRPIX3")
    header.pop("CUNIT3")
    header.pop("NAXIS3")

    try:
        header.pop("PC3_1")
        header.pop("PC3_2")
        header.pop("PC1_3")
        header.pop("PC2_3")
        header.pop("PC3_3")
    except:
        pass

    return header


def create_vel_array(galaxy, cube, spec_res=10, savepath=None):
    """
    Creates the velocity array corresponding to the spectral axis
    of the cube in km/s.

    Parameters
    ----------
    galaxy : string
        Name of the galaxy for which the velocity array is created.
    cube : FITS image
        Containing the 3D data cube and corresponding header.
    spec_res : int or float, optional
        The spectral resolution of the data cube in km/s. The default is 10.
    savepath : string, optional
        The location in which the array is saved. The array is only saved if
        this is provided. The default is None.

    Raises
    ------
    KeyError
        Raised if the units of the cube spectral axis (as described in the header)
        are something other than velocity or frequency.

    Returns
    -------
    vel_array : NumPy array
        1D array containing the velocities corresponding to each channel of the
        spectral axis of the cube provided.
    vel_narray : NumPy array
        Similar to "vel_array" but tiled to match the cube dimensions.
    v_step : float
        The spectral resolution of the velocity array in km/s.

    """

    v_ref = cube.header["CRPIX3"]  # Location of the reference channel

    if (
        cube.header["CTYPE3"] == "VRAD"
        or cube.header["CTYPE3"] == "VELOCITY"
        or cube.header["CTYPE3"] == "VOPT"
        or cube.header["CTYPE3"] == "VOPT-W2W"
    ):
        if cube.header["CUNIT3"] != "km s-1":
            v_val = (
                cube.header["CRVAL3"] / 1000
            )  # Velocity in the reference channel, m/s to km/s
            v_step = (
                cube.header["CDELT3"] / 1000
            )  # Velocity step in each channel, m/s to km/s
        else:
            v_val = cube.header["CRVAL3"]  # Velocity in the reference channel
            v_step = cube.header["CDELT3"]  # Velocity step in each channel

    # If the spectral axis of the cube is the frequency, calculate the relevant
    # velocity information

    # NOTE THAT THIS ASSUMES THE AXIS IS IN GHz AND WE ARE OBSERVING THE CO(2-1)
    # LINE !!!!!!!!!!!

    elif cube.header["CTYPE3"] == "FREQ":
        v_val = 299792.458 * (1 - (cube.header["CRVAL3"] / 1e9) / 230.538000)
        v_shift = 299792.458 * (
            1 - ((cube.header["CRVAL3"] + cube.header["CDELT3"]) / 1e9) / 230.538000
        )
        v_step = -(v_val - v_shift)
    else:
        raise KeyError("Pipeline cannot deal with these units yet.")

    # Construct the velocity arrays (keep in mind that fits-files are 1 indexed)
    # vel_array = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1 + self.galaxy.start) * v_step + v_val
    vel_array = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1) * v_step + v_val
    vel_narray = np.tile(
        vel_array, (len(cube.data[0, 0, :]), len(cube.data[0, :, 0]), 1)
    ).transpose()

    if savepath:
        if spec_res == 10:
            np.save(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/10kms/"
                + galaxy
                + "_vel_array.npy",
                vel_array,
            )
            if not os.path.exists(savepath + "by_product/spectrum"):
                os.makedirs(savepath + "by_product/spectrum", exist_ok=True)
            if not os.path.exists(savepath + "by_product/spectrum/10kms"):
                os.makedirs(savepath + "by_product/spectrum/10kms", exist_ok=True)
            np.save(
                savepath + "by_product/spectrum/10kms/" + galaxy + "_vel_array.npy",
                vel_array,
            )
        elif spec_res == 30:
            np.save(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/30kms/"
                + galaxy
                + "_vel_array.npy",
                vel_array,
            )
            if not os.path.exists(savepath + "by_product/spectrum"):
                os.makedirs(savepath + "by_product/spectrum", exist_ok=True)
            if not os.path.exists(savepath + "by_product/spectrum/30kms"):
                os.makedirs(savepath + "by_product/spectrum/30kms", exist_ok=True)
            np.save(
                savepath + "by_product/spectrum/30kms/" + galaxy + "_vel_array.npy",
                vel_array,
            )

    return vel_array, vel_narray, v_step


def calc_moms(
    cube,
    galaxy,
    glob_cat,
    spec_res=10,
    savepath=None,
    units="K km/s",
    alpha_co=4.35,
    R21=0.7,
    subtract_vsys=False,
):
    """
    Calculates the moment zero, one, and two maps from the 3D data cube.

    Parameters
    ----------
    cube : FITS image
        Contains the 3D data cube and corresponding header.
    galaxy : string
        Name of the galaxy for which the moment maps are created.
    glob_cat : string
        File name of the KILOGAS global catalogue containing the global galaxy parameters.
    spec_res : foat, optional
        Spectral resolution of the data cube in km/s. The default is 10.
    savepath : string, optional
        Path to the directory where the moment maps should be saved. The maps
        are only saved if this is provided. The default is None.
    units : string, optional
        The units in which the moment zero map should be calculated. Options are
        'Msol pc-2', 'Msol/pix', 'K km/s pc^2', and 'K km/s'. The default is 'K km/s'.
    alpha_co : TYPE, optional
        DESCRIPTION. The default is 4.35.
    R21 : TYPE, optional
        DESCRIPTION. The default is 0.7.
    subtract_vsys : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mom0_hdu : TYPE
        DESCRIPTION.
    mom1_hdu : TYPE
        DESCRIPTION.
    mom2_hdu : TYPE
        DESCRIPTION.

    """

    # Calculate the velocities of the cube spectral axis in km/s
    vel_array, vel_narray, dv = create_vel_array(
        galaxy, cube, spec_res=spec_res, savepath=savepath
    )

    # Calculate the moment zero
    mom0 = np.nansum((cube.data * dv), axis=0)

    # Set any negative values (as a result of clipping) to NaN
    mom0[mom0 < 0] = np.nan

    # Set redshift parameters needed for calculations of additional moment zero
    # maps with physical units

    # Get the galaxy redshift from the KILOGAS global catalogue
    glob_tab = fits.open(glob_cat)[1]
    z = glob_tab.data["Z"][glob_tab.data["KGAS_ID"] == int(galaxy.split("KGAS")[1])][0]

    # Define the cosmology we work in
    # NOTE THAT THIS IS HARDCODED TO ASSUME A STANDARD LCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Calculate the # of pc per pixel within the assumed cosmology
    pc_to_pix = (
        cube.header["CDELT2"] * cosmo.kpc_proper_per_arcmin(z).value * 60 * 1000
    ) ** 2

    # Calculated the moment map in the requested physical units.

    # NOTE THAT THIS CURRENTLY DOES NOT TOLERATE ANY TYPOS AND DOES NOT THROW
    # AN ERROR IF UNITS ARE PROVIDED WHICH ARE NOT AMONG THE THREE OPTIONS BELOW.
    # IN THIS CASE THE DEFAULT UNITS OF 'K km/s' WILL BE RETURNED.

    if units == "Msol pc-2":
        mom0 *= alpha_co
        mom0 /= R21
        mom0 *= 1 + z

        # Correct for inclination by multiplying by b/a
        # NOTE: This is currently hardcoded until the inclinations are stored somewhere more permanently.
        inc_table = fits.open("KGAS_global_master.fits")[1]
        ba = inc_table.data["ba"][
            inc_table.data["KGID"] == int(galaxy.split("KGAS")[1])
        ]
        mom0 *= ba

    elif units == "Msol/pix":
        mom0 *= alpha_co
        mom0 /= R21
        mom0 *= 1 + z
        mom0 *= pc_to_pix

    elif units == "K km/s pc^2":
        mom0 *= pc_to_pix
        mom0 *= 1 + z

    # Set any resulting zeros or unphysical values to NaN
    mom0[np.isinf(mom0)] = np.nan
    mom0[mom0 == 0] = np.nan

    # Calculate the moment one map
    mom1 = np.nansum(cube.data * vel_narray, axis=0) / np.nansum(cube.data, axis=0)

    # Calculate the moment two map
    mom2 = np.sqrt(
        np.nansum(abs(cube.data) * (vel_narray - mom1) ** 2, axis=0)
        / np.nansum(abs(cube.data), axis=0)
    )

    # Estimate the systemic velocity from the spatial inner part of the cube 
    
    if subtract_vsys:
        inner_cube = innersquare(mom1)
        vsys = np.nanmean(inner_cube)
        mom1 -= vsys

        # Uncomment two lines below to check if "inner_cube" is reasonable
        # from matplotlib import pyplot as plt
        # plt.imshow(inner_cube)

        # If this option is going to be used, vsys should be added to the header
        # and/or saved to a text file.

    # Create 2D headers for the moment maps from the 3D cube header
    moment_header = new_header(cube.header)

    # Create FITS files for the moment maps
    mom0_hdu = fits.PrimaryHDU(mom0, moment_header)
    mom1_hdu = fits.PrimaryHDU(mom1, moment_header)
    mom2_hdu = fits.PrimaryHDU(mom2, moment_header)

    # Change or add any additional keywords to the headers
    if units == "K km/s pc^2":
        mom0_hdu.header["BTYPE"] = "Lco"
        mom0_hdu.header.comments["BTYPE"] = "CO luminosity"
        mom0_hdu.header["BUNIT"] = units
        mom0_hdu.header.comments["BUNIT"] = ""
    elif units == "Msol pc-2":
        mom0_hdu.header["BTYPE"] = "mmol pc^-2"
        mom0_hdu.header.comments["BTYPE"] = "Molecular gas mass surface density"
        mom0_hdu.header["BUNIT"] = "Msun pc^-2"
        mom0_hdu.header.comments["BUNIT"] = ""
    elif units == "Msol/pix":
        mom0_hdu.header["BTYPE"] = "mmol_pix"
        mom0_hdu.header.comments["BTYPE"] = "Molecular gas mass in pixel"
        mom0_hdu.header["BUNIT"] = "Msun"
        mom0_hdu.header.comments["BUNIT"] = ""
    else:
        mom0_hdu.header["BTYPE"] = "Ico"
        mom0_hdu.header.comments["BTYPE"] = "CO surface brightness"
        mom0_hdu.header["BUNIT"] = units
        mom0_hdu.header.comments["BUNIT"] = ""

    mom1_hdu.header["BTYPE"] = "co_vel"
    mom2_hdu.header["BTYPE"] = "co_obs_lw"

    mom1_hdu.header.comments["BTYPE"] = "Absolute CO velocity"
    mom2_hdu.header.comments["BTYPE"] = "Observed line of sight CO line width"

    mom1_hdu.header["BUNIT"] = "km/s"
    mom1_hdu.header.comments["BUNIT"] = ""
    mom2_hdu.header["BUNIT"] = "km/s"
    mom2_hdu.header.comments["BUNIT"] = ""

    if savepath:
        if spec_res == 10:
            if not os.path.exists(savepath + "by_product/moment_maps"):
                os.makedirs(savepath + "by_product/moment_maps", exist_ok=True)
            if not os.path.exists(savepath + "by_product/moment_maps/10kms"):
                os.makedirs(savepath + "by_product/moment_maps/10kms", exist_ok=True)
            if units == "K km/s":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_Ico_K_kms-1.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_Ico_K_kms-1.fits",
                    overwrite=True,
                )
            elif units == "K km/s pc^2":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2.fits",
                    overwrite=True,
                )
            elif units == "Msol pc-2":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mmol_pc-2.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mmol_pc-2.fits",
                    overwrite=True,
                )
            elif units == "Msol/pix":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mmol_pix-1.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mmol_pix-1.fits",
                    overwrite=True,
                )

            mom1_hdu.writeto(
                savepath + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_mom1.fits",
                overwrite=True,
            )
            mom1_hdu.writeto(
                savepath + "by_product/moment_maps/10kms/" + galaxy + "_mom1.fits",
                overwrite=True,
            )

            mom2_hdu.writeto(
                savepath + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_mom2.fits",
                overwrite=True,
            )
            mom2_hdu.writeto(
                savepath + "by_product/moment_maps/10kms/" + galaxy + "_mom2.fits",
                overwrite=True,
            )

        elif spec_res == 30:
            if not os.path.exists(savepath + "by_product/moment_maps"):
                os.makedirs(savepath + "by_product/moment_maps", exist_ok=True)
            if not os.path.exists(savepath + "by_product/moment_maps/30kms"):
                os.makedirs(savepath + "by_product/moment_maps/30kms", exist_ok=True)
            if units == "K km/s":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_Ico_K_kms-1.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_Ico_K_kms-1.fits",
                    overwrite=True,
                )
            elif units == "K km/s pc^2":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2.fits",
                    overwrite=True,
                )
            elif units == "Msol pc-2":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mmol_pc-2.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mmol_pc-2.fits",
                    overwrite=True,
                )
            elif units == "Msol/pix":
                mom0_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mmol_pix-1.fits",
                    overwrite=True,
                )
                mom0_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mmol_pix-1.fits",
                    overwrite=True,
                )

            mom1_hdu.writeto(
                savepath + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_mom1.fits",
                overwrite=True,
            )
            mom1_hdu.writeto(
                savepath + "by_product/moment_maps/30kms/" + galaxy + "_mom1.fits",
                overwrite=True,
            )

            mom2_hdu.writeto(
                savepath + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_mom2.fits",
                overwrite=True,
            )
            mom2_hdu.writeto(
                savepath + "by_product/moment_maps/30kms/" + galaxy + "_mom2.fits",
                overwrite=True,
            )

    return mom0_hdu, mom1_hdu, mom2_hdu


def calc_uncs(
    cube,
    path,
    galaxy,
    glob_cat,
    savepath,
    ifu_match,
    spec_res=10,
    units="K km/s",
    alpha_co=4.35,
    R21=0.7,
    lw=30,
    ul=3,
    pb_thresh=40,
    detection=True,
    clipped_cube=None,
):
    """
    Create uncertainty maps OR upper limit maps corresponding to the moment zero map.

    Parameters
    ----------
    cube : FITS image
        Contains the data cube and corresponding header..
    path : string
        Directory in which the products are stored.
    galaxy : string
        Name of the galaxy for which the map is created.
    glob_cat : string
        Filename of the KILOGAS global catalogue.
    savepath : string
        Path to the directory in which the maps should be saved.
    ifu_match : bool
        True if the data cube is matched to the IFU resolution and False if it
        has the original resolution.
    spec_res : float, optional
        Spectral resolution of the data cube in km/s. Can be either 10 or 30.
        The default is 10.
    units : string, optional
        Units of the map that is created. Options are 'Msol pc-2', 'Msol/pix',
        'K km/s pc^2', and 'K km/s'.  The default is 'K km/s'.
    alpha_co : float, optional
        Value of the CO-to-H2 conversion factor to be used, in Msol/​(K km/s pc^2).
        The default is 4.35.
    R21 : float, optional
        CO(2-1)/CO(1-0) intensity ratio. The default is 0.7.
    lw : float, optional
        Assumed line width for the upper limit maps in km/s. The default is 30.
    ul : float, optional
        The 'ul'sigma upper limit will be calculated. The default is 3.
    pb_thresh : float, optional
        Percentage of the max. primary beam response below which the maps will
        be masked. The default is 40.
    detection : bool, optional
        True if the galaxy is detected, in this case an uncertainty map will be
        produced. False if the galaxy is not detected, in this case an upper
        limit map will be produced (note that this is historical and not the best
        name for this variable, should be changed). The default is True.
    clipped_cube : FITS file, optional
        Contains a clipped version of the cube and corresponding header. If
        provided, the rms that was used to clip the cube is adopted for the creation
        of the uncertainty or upper limit map. The default is None.

    Returns
    -------
    None.

    """

    # Calculate the number of channels by converting the cube into a boolean
    cube_bool = cube.data.copy()

    # Set any nans to 0 to avoid them being converted to True
    cube_bool[cube_bool != cube_bool] = 0
    cube_bool = cube_bool.astype("bool")

    N_map = np.sum(cube_bool, axis=0)

    # Find the corresponding pb uncorrected cube
    if ifu_match == True:
        try:
            path_pbcorr = (
                path
                + galaxy
                + "/"
                + galaxy
                + "_co2-1_"
                + str(spec_res)
                + ".0kmps_12m.image.pbcor.ifumatched.fits"
            )
            path_uncorr = (
                path
                + galaxy
                + "/"
                + galaxy
                + "_co2-1_"
                + str(spec_res)
                + ".0kmps_12m.image.ifumatched.fits"
            )
            cube_pb_corr = fits.open(path_pbcorr)[0]
        except:
            try:
                path_pbcorr = (
                    path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(spec_res)
                    + ".0kmps_7m+12m.image.pbcor.ifumatched.fits"
                )
                path_uncorr = (
                    path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(spec_res)
                    + ".0kmps_7m+12m.image.ifumatched.fits"
                )
                cube_pb_corr = fits.open(path_pbcorr)[0]
            except:
                try:
                    path_pbcorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_12m.contsub.image.pbcor.ifumatched.fits"
                    )
                    path_uncorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_12m.contsub.image.ifumatched.fits"
                    )
                    cube_pb_corr = fits.open(path_pbcorr)[0]
                except:
                    path_pbcorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_7m+12m.contsub.image.pbcor.ifumatched.fits"
                    )
                    path_uncorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_7m+12m.contsub.image.ifumatched.fits"
                    )
                    cube_pb_corr = fits.open(path_pbcorr)[0]
    else:
        try:
            path_pbcorr = (
                path
                + galaxy
                + "/"
                + galaxy
                + "_co2-1_"
                + str(spec_res)
                + ".0kmps_12m.image.pbcor.fits"
            )
            path_uncorr = (
                path
                + galaxy
                + "/"
                + galaxy
                + "_co2-1_"
                + str(spec_res)
                + ".0kmps_12m.image.fits"
            )
            cube_pb_corr = fits.open(path_pbcorr)[0]
        except:
            try:
                path_pbcorr = (
                    path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(spec_res)
                    + ".0kmps_7m+12m.image.pbcor.fits"
                )
                path_uncorr = (
                    path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(spec_res)
                    + ".0kmps_7m+12m.image.fits"
                )
                cube_pb_corr = fits.open(path_pbcorr)[0]
            except:
                try:
                    path_pbcorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_12m.contsub.image.pbcor.fits"
                    )
                    path_uncorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_12m.contsub.image.fits"
                    )
                    cube_pb_corr = fits.open(path_pbcorr)[0]
                except:
                    path_pbcorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_7m+12m.contsub.image.pbcor.fits"
                    )
                    path_uncorr = (
                        path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(spec_res)
                        + ".0kmps_7m+12m.contsub.image.fits"
                    )
                    cube_pb_corr = fits.open(path_pbcorr)[0]

    # Use the pb uncorrected cube to calculate the pb response cube
    cube_uncorr = fits.open(path_uncorr)[0]
    pb_cube = cube_pb_corr.copy()
    pb_cube.data = cube_uncorr.data / cube_pb_corr.data
    pb_cube.data[cube_bool.data != cube_bool.data] = np.nan

    # If a clipped cube is provided, use the rms used for clipping to create the
    # noise cube. Otherwise calculate the rms from the spatial inner square of the cube
    if clipped_cube is not None:
        noise_cube = clipped_cube.header["CLIP_RMS"] / pb_cube.data
    else:
        inner_square = innersquare(cube_uncorr.data)
        noise_cube = np.nanstd(inner_square) / pb_cube.data

    # Collapse the noise cube to create a 2D noise map
    noise_map = np.nanmedian(noise_cube, axis=0)

    # If upper limits are being calculated, mask the noise map beyond a given
    # primary beam response threshold
    if not detection:
        noise_map[np.nanmedian(pb_cube.data, axis=0) < pb_thresh / 100] = np.nan

    # Similarly to moment map creation, set redshift parameters needed for physical unit calculations

    # NOTE: COULD MAKE THIS A SEPARATE FUNCTION WITH OPTIONAL COSMO PARAMETERS

    glob_tab = fits.open(glob_cat)[1]
    z = glob_tab.data["Z"][glob_tab.data["KGAS_ID"] == int(galaxy.split("KGAS")[1])][0]
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    pc_to_pix = (
        cube.header["CDELT2"] * cosmo.kpc_proper_per_arcmin(z).value * 60 * 1000
    ) ** 2

    # If uncertainties are being calculated (rather than upper limits), create
    # the moment zero uncertainty map
    if detection:
        if cube.header["CUNIT3"] != "km s-1":
            mom0_uncertainty = (
                noise_map * np.sqrt(N_map) * abs(cube.header["CDELT3"] / 1000)
            )
        else:
            mom0_uncertainty = noise_map * np.sqrt(N_map) * abs(cube.header["CDELT3"])

    # If upper limits are being estimated, create upper limit maps for the moment zero
    elif cube.header["CUNIT3"] != "km s-1":
        mom0_uncertainty = (
            noise_map
            * abs(cube.header["CDELT3"])
            / 1000
            * ul
            * np.sqrt(lw / abs(cube.header["CDELT3"] / 1000))
        )
    else:
        mom0_uncertainty = (
            noise_map
            * abs(cube.header["CDELT3"])
            * ul
            * np.sqrt(lw / abs(cube.header["CDELT3"]))
        )

    # Set any non-physical pixels to NaN
    mom0_uncertainty[np.isinf(mom0_uncertainty)] = np.nan
    mom0_uncertainty[mom0_uncertainty <= 0] = np.nan

    # Convert the uncertainty or upper limit map to different units if requested.
    # If "savepath" is provided the map is saved in this directory.

    # NOTE THAT SIMILARLY TO THE MOMENT MAP CREATION THERE IS NO FAILSAFE FOR TYPOS
    # OR WRONGLY PROVIDED UNITS SO ANY OF THAT WILL DEFAULT TO THE STANDARD UNITS
    # OF 'K km s^-1'

    if units == "Msol pc-2":
        mom0_hdu, _, _ = calc_moms(
            cube, galaxy, glob_cat, spec_res=spec_res, savepath=None, units="Msol pc-2"
        )
        mom0_uncertainty *= alpha_co
        mom0_uncertainty /= R21
        mom0_uncertainty *= 1 + z

        # Correct for inclination by multiplying by b/a
        inc_table = fits.open("KGAS_global_master.fits")[1]
        ba = inc_table.data["ba"][
            inc_table.data["KGID"] == int(galaxy.split("KGAS")[1])
        ]
        mom0_uncertainty *= ba

        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header["BTYPE"] = "mmol pc^-2 error"
        mom0_uncertainty_hdu.header.comments[
            "BTYPE"
        ] = "Error mol. gas mass surf. dens."
        mom0_hdu.header["BUNIT"] = "Msun pc-2"
        mom0_hdu.header.comments["BUNIT"] = ""

        if detection:
            if spec_res == 10:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mmol_pc-2_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mmol_pc-2_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mmol_pc-2_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mmol_pc-2_err.fits",
                    overwrite=True,
                )
        elif spec_res == 10:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/10kms/"
                + galaxy
                + "_mmol_pc-2_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/10kms/"
                + galaxy
                + "_mmol_pc-2_ul.fits",
                overwrite=True,
            )
        elif spec_res == 30:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/30kms/"
                + galaxy
                + "_mmol_pc-2_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/30kms/"
                + galaxy
                + "_mmol_pc-2_ul.fits",
                overwrite=True,
            )

    elif units == "Msol/pix":
        mom0_hdu, _, _ = calc_moms(
            cube, galaxy, glob_cat, spec_res=spec_res, savepath=None, units="Msol/pix"
        )

        mom0_uncertainty *= alpha_co
        mom0_uncertainty /= R21
        mom0_uncertainty *= 1 + z
        mom0_uncertainty *= pc_to_pix

        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header["BTYPE"] = "mmol error"
        mom0_uncertainty_hdu.header.comments["BTYPE"] = "Error mol. gas mass in pixel"
        mom0_hdu.header["BUNIT"] = "Msun pix^-1"
        mom0_hdu.header.comments["BUNIT"] = ""

        if detection:
            if spec_res == 10:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mmol_pix-1_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mmol_pix-1_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mmol_pix-1_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mmol_pix-1_err.fits",
                    overwrite=True,
                )
        elif spec_res == 10:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/10kms/"
                + galaxy
                + "_mmol_pix-1_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/10kms/"
                + galaxy
                + "_mmol_pix-1_ul.fits",
                overwrite=True,
            )
        elif spec_res == 30:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/30kms/"
                + galaxy
                + "_mmol_pix-1_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/30kms/"
                + galaxy
                + "_mmol_pix-1_ul.fits",
                overwrite=True,
            )

    elif units == "K km/s pc^2":
        (
            mom0_hdu,
            _,
            _,
        ) = calc_moms(cube, galaxy, glob_cat, spec_res=spec_res, units="K km/s pc^2")

        mom0_uncertainty *= pc_to_pix
        mom0_uncertainty *= 1 + z

        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header["BTYPE"] = "Lco error"
        mom0_uncertainty_hdu.header.comments["BTYPE"] = "Error in CO luminosity"
        mom0_hdu.header["BUNIT"] = "K km s^-1 pc^2"
        mom0_hdu.header.comments["BUNIT"] = ""

        if detection:
            if spec_res == 10:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_Lco_K_kms-1_pc2_err.fits",
                    overwrite=True,
                )
        elif spec_res == 10:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/10kms/"
                + galaxy
                + "_Lco_K_kms-1_pc2_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/10kms/"
                + galaxy
                + "_Lco_K_kms-1_pc2_ul.fits",
                overwrite=True,
            )
        elif spec_res == 30:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/30kms/"
                + galaxy
                + "_Lco_K_kms-1_pc2_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/30kms/"
                + galaxy
                + "_Lco_K_kms-1_pc2_ul.fits",
                overwrite=True,
            )

    else:
        mom0_hdu, mom1_hdu, mom2_hdu = calc_moms(cube, galaxy, glob_cat)

        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header["BTYPE"] = "Ico error"
        mom0_uncertainty_hdu.header.comments["BTYPE"] = "Error in CO SB"
        mom0_hdu.header["BUNIT"] = "K km s^-1"
        mom0_hdu.header.comments["BUNIT"] = ""

        if detection:
            if spec_res == 10:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_Ico_K_kms-1_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_Ico_K_kms-1_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_Ico_K_kms-1_err.fits",
                    overwrite=True,
                )
                mom0_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_Ico_K_kms-1_err.fits",
                    overwrite=True,
                )
        elif spec_res == 10:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/10kms/"
                + galaxy
                + "_Ico_K_kms-1_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/10kms/"
                + galaxy
                + "_Ico_K_kms-1_ul.fits",
                overwrite=True,
            )
        elif spec_res == 30:
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_galaxy/"
                + galaxy
                + "/30kms/"
                + galaxy
                + "_Ico_K_kms-1_ul.fits",
                overwrite=True,
            )
            mom0_uncertainty_hdu.writeto(
                savepath
                + "by_product/moment_maps/30kms/"
                + galaxy
                + "_Ico_K_kms-1_ul.fits",
                overwrite=True,
            )

        if detection:
            SN_map = mom0_hdu.data / mom0_uncertainty
            SN_hdu = fits.PrimaryHDU(SN_map, mom0_hdu.header)
            SN_hdu.header.pop("BUNIT")

            if spec_res == 10:
                SN_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mom0_SN.fits",
                    overwrite=True,
                )
                SN_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mom0_SN.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                SN_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mom0_SN.fits",
                    overwrite=True,
                )
                SN_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mom0_SN.fits",
                    overwrite=True,
                )

            if cube.header["CUNIT3"] != "km s-1":
                mom1_uncertainty = (
                    N_map * abs(cube.header["CDELT3"] / 1000) / (2 * np.sqrt(3))
                ) * (
                    mom0_uncertainty / mom0_hdu.data
                )  # Eqn 15 doc. Chris
            else:
                mom1_uncertainty = (
                    N_map * abs(cube.header["CDELT3"]) / (2 * np.sqrt(3))
                ) * (
                    mom0_uncertainty / mom0_hdu.data
                )  # Eqn 15 doc. Chris
            mom1_uncertainty_hdu = fits.PrimaryHDU(mom1_uncertainty, mom1_hdu.header)
            mom1_uncertainty_hdu.header["BTYPE"] = "velocity error"

            if cube.header["CUNIT3"] != "km s-1":
                mom2_uncertainty = (
                    (
                        (N_map * abs(cube.header["CDELT3"] / 1000)) ** 2
                        / (8 * np.sqrt(5))
                    )
                    * (mom0_uncertainty / mom0_hdu.data)
                    * (mom2_hdu.data) ** -1
                )  # Eqn 30 doc. Chris
            else:
                mom2_uncertainty = (
                    ((N_map * abs(cube.header["CDELT3"])) ** 2 / (8 * np.sqrt(5)))
                    * (mom0_uncertainty / mom0_hdu.data)
                    * (mom2_hdu.data) ** -1
                )  # Eqn 30 doc. Chris

            mom2_uncertainty_hdu = fits.PrimaryHDU(mom2_uncertainty, mom2_hdu.header)
            mom2_uncertainty_hdu.header["BTYPE"] = "obs. line width error"

            if spec_res == 10:
                mom1_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mom1_err.fits",
                    overwrite=True,
                )
                mom1_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mom1_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom1_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mom1_err.fits",
                    overwrite=True,
                )
                mom1_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mom1_err.fits",
                    overwrite=True,
                )

            if spec_res == 10:
                mom2_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/10kms/"
                    + galaxy
                    + "_mom2_err.fits",
                    overwrite=True,
                )
                mom2_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/10kms/"
                    + galaxy
                    + "_mom2_err.fits",
                    overwrite=True,
                )
            elif spec_res == 30:
                mom2_uncertainty_hdu.writeto(
                    savepath
                    + "by_galaxy/"
                    + galaxy
                    + "/30kms/"
                    + galaxy
                    + "_mom2_err.fits",
                    overwrite=True,
                )
                mom2_uncertainty_hdu.writeto(
                    savepath
                    + "by_product/moment_maps/30kms/"
                    + galaxy
                    + "_mom2_err.fits",
                    overwrite=True,
                )


def calc_peak_t(cube, galaxy, spec_res=10, savepath=None):
    """
    Create a peak temperature map.

    Parameters
    ----------
    cube : FITS image
        Contains the data cube and corresponding header.
    galaxy : string
        KILOGAS name of the galaxy for which the map is made.
    spec_res : float, optional
        Spectral resolution of the data cube in km/s. Can be 10 or 30.
        The default is 10.
    savepath : string, optional
        Path to the directory where the maps should be stored. The maps are
        only stored if 'savepath' is provided. The default is None.

    Returns
    -------
    None.

    """

    peak_temp = np.nanmax(cube.data, axis=0)

    peak_temp_hdu = fits.PrimaryHDU(peak_temp, new_header(cube.header))
    peak_temp_hdu.header["BTYPE"] = "Peak temperature"
    peak_temp_hdu.header["BUNIT"] = "K"
    peak_temp_hdu.header.comments["BUNIT"] = ""

    if spec_res == 10:
        peak_temp_hdu.writeto(
            savepath + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_peak_temp_k.fits",
            overwrite=True,
        )
        peak_temp_hdu.writeto(
            savepath + "by_product/moment_maps/10kms/" + galaxy + "_peak_temp_k.fits",
            overwrite=True,
        )

    elif spec_res == 30:
        peak_temp_hdu.writeto(
            savepath + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_peak_temp_k.fits",
            overwrite=True,
        )
        peak_temp_hdu.writeto(
            savepath + "by_product/moment_maps/30kms/" + galaxy + "_peak_temp_k.fits",
            overwrite=True,
        )


def perform_moment_creation(
    path,
    data_path,
    detections,
    non_detections,
    glob_cat,
    ifu_match,
    spec_res=10,
    pb_thresh=40,
    overwrite=True,
):
    for galaxy in detections:
        print(galaxy)

        # Ensure output directories exist
        if spec_res == 10:
            os.makedirs(path + "by_galaxy/" + galaxy + "/10kms", exist_ok=True)
            os.makedirs(path + "by_product/moment_maps/10kms", exist_ok=True)
        elif spec_res == 30:
            os.makedirs(path + "by_galaxy/" + galaxy + "/30kms", exist_ok=True)
            os.makedirs(path + "by_product/moment_maps/30kms", exist_ok=True)

        # Skip if already exists
        if not overwrite:
            check = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + galaxy + '_Ico_K_kms-1.fits'
            if os.path.exists(check):
                print(f'  Skipping {galaxy} (exists, overwrite=False)')
                continue

        # ── Open cubes ONCE ──
        if spec_res == 10:
            cube = glob(path + "by_galaxy/" + galaxy + "/10kms/*clipped_cube.fits")
        elif spec_res == 30:
            cube = glob(path + "by_galaxy/" + galaxy + "/30kms/*clipped_cube.fits")

        if ifu_match:
            cube_raw = glob(data_path + galaxy + "/*" + str(spec_res) + "*.image.ifumatched.fits")
        else:
            cube_raw = glob(data_path + galaxy + "/*" + str(spec_res) + "*.image.fits")

        cube_fits = fits.open(cube[0])[0]
        cube_raw_fits = fits.open(cube_raw[0])[0]

        # ── Create moment maps ONCE per unit, cache results ──
        all_units = ["K km/s", "K km/s pc^2", "Msol pc-2", "Msol/pix"]
        mom0_cache = {}

        for units in all_units:
            mom0_hdu, mom1_hdu, mom2_hdu = calc_moms(
                cube_fits, galaxy, glob_cat=glob_cat,
                spec_res=spec_res, savepath=path,
                units=units, alpha_co=4.35, R21=0.7,
            )
            mom0_cache[units] = (mom0_hdu, mom1_hdu, mom2_hdu)

        calc_peak_t(cube_fits, galaxy, spec_res=spec_res, savepath=path)

        # ── Create uncertainty maps, passing cached mom0 ──
        for units in all_units:
            # Error maps (detection=True)
            calc_uncs(
                cube_fits, data_path, galaxy,
                glob_cat=glob_cat, spec_res=spec_res,
                savepath=path, ifu_match=ifu_match,
                clipped_cube=cube_fits,
                units=units, alpha_co=4.35, R21=0.7,
                detection=True,
            )

        # ── Upper limit maps ──
        for units in all_units:
            calc_uncs(
                cube_raw_fits, data_path, galaxy,
                glob_cat=glob_cat, spec_res=spec_res,
                savepath=path, ifu_match=ifu_match,
                clipped_cube=cube_fits,
                units=units, alpha_co=4.35, R21=0.7,
                detection=False, lw=30, ul=3, pb_thresh=pb_thresh,
            )

    for galaxy in non_detections:
        print(galaxy)

        # Ensure output directories exist
        if spec_res == 10:
            os.makedirs(path + "by_galaxy/" + galaxy + "/10kms", exist_ok=True)
            os.makedirs(path + "by_product/moment_maps/10kms", exist_ok=True)
        elif spec_res == 30:
            os.makedirs(path + "by_galaxy/" + galaxy + "/30kms", exist_ok=True)
            os.makedirs(path + "by_product/moment_maps/30kms", exist_ok=True)

        if not overwrite:
            check = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + galaxy + '_Ico_K_kms-1_ul.fits'
            if os.path.exists(check):
                print(f'  Skipping {galaxy} (exists, overwrite=False)')
                continue

        if ifu_match:
            cube = glob(data_path + galaxy + "/*" + str(spec_res) + "*.image.ifumatched.fits")
        elif not ifu_match:
            cube = glob(data_path + galaxy + "/*" + str(spec_res) + "*.image.fits")

        try:
            cube_fits = fits.open(cube[0])[0]
        except:
            continue

        for units in all_units:
            calc_uncs(
                cube_fits, data_path, galaxy,
                glob_cat=glob_cat, spec_res=spec_res,
                savepath=path, ifu_match=ifu_match,
                units=units, alpha_co=4.35, R21=0.7,
                detection=False, lw=30, ul=3, pb_thresh=pb_thresh,
            )


if __name__ == "__main__":
    
    # NOTE: all hardcoded for now as it isn't really used, but can improve
    # to take user input.
    
    spec_res = 10
    ifu_match = False
    pb_thresh = 40    
    
    path = "/arc/projects/KILOGAS/products/v1.1/matched/"
    data_path = "/arc/projects/KILOGAS/cubes/v1.0/matched/"

    targets = [
        "KGAS73",
        "KGAS128",
        "KGAS184",
        "KGAS255",
        "KGAS262",
        "KGAS288",
        "KGAS328",
        "KGAS371",
        "KGAS397",
    ]
    non_detections = [
        "KGAS73",
        "KGAS128",
        "KGAS184",
        "KGAS255",
        "KGAS262",
        "KGAS288",
        "KGAS328",
        "KGAS371",
        "KGAS397",
    ]
    detections = []

    glob_cat = "KILOGAS_global_catalog_FWHM.fits"

    perform_moment_creation(
        path=path,
        data_path=data_path,
        detections=detections,
        non_detections=non_detections,
        glob_cat=glob_cat,
        spec_res=spec_res,
        ifu_match=ifu_match,
        pb_thresh=pb_thresh,
    )
    
