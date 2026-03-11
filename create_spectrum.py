from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture
from glob import glob
import os
from create_moments import create_vel_array
import pandas as pd


def gauss(x, a, x0, sigma):
    """
    Calculate a Gaussian function according to the parameters provided.

    Parameters
    ----------
    x : NumPy array
        1D array containing the x-values over which the Gaussian is defined..
    a : float
        The amplitude.
    x0 : float
        The centre/median.
    sigma : float
        Standard deviation.

    Returns
    -------
    gauss : NumPy array
        y-values for each value of x corresponding to the Gaussian function calculated.

    """

    gauss = a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    return gauss


def calc_beam_area(bmaj, bmin, cellsize):
    """

    Parameters
    ----------
    bmaj : float
        Beam major axis length.
    bmin : float
        Beam minor axis length.
    cellsize : float
        The resolution (pixel size) of the cube in the same units as "bmaj" and "bmin" / pixel.

    Returns
    -------
    beam_area : float
        The beam area in the same units as "bmaj" and "bmin".

    """

    beam_area = (np.pi * (bmaj / cellsize) * (bmin / cellsize)) / (4 * np.log(2))

    return beam_area


def brightness_temp_to_flux_dens(T, bmaj, bmin, nu=230.538):
    """
    Calculates the flux density corresponding to the brightness temperature
    provided at a given frequency. The default frequency corresponds to the
    CO(2-1) transition.

    Parameters
    ----------
    T : float or NumPy array
        Brightness temperature or an array of brightness temperatures.
    bmaj : float
        Synthesised beam major axis length in arcseconds.
    bmin : float
        Synthesised beam minor axis length in arcseconds..
    nu : float, optional
        Frequency at which the flux density is calculated. The default is 230.538.

    Returns
    -------
    flux_dens : float or NumPy array
        The flux density/densities corresponding to the input brightness temperature.

    """

    flux_dens = T * nu**2 * bmaj * bmin / 1.222e3

    return flux_dens


def make_spectrum(
    cube,
    galaxy,
    start,
    stop,
    path,
    glob_cat,
    extra_chans=10,
    non_det=False,
    spec_res=10,
):
    """
    Extract the spectrum of the CO line. If the galaxy is detected, it is extracted
    from the detected area. If no CO is detected, the spectrum is extracted from
    within r50.

    Parameters
    ----------
    cube : FITS image file
        Contains the 3D data cube and corresponding header.
    galaxy : string
        The name of the galaxy.
    start : float
        The first channel in which CO is detected.
    stop : float
        The last channel in which CO is detected.
    path : string
        Path to the directory in which the clip mask is saved and where the resulting
        spectrum will be stored.
    glob_cat : string
        Full path (incl. fiename) to where the table with global parameters is stored.
    extra_chans : float, optional
        The number of (empty) channels to be added on either side of the spectrum. The default is 10.
    non_det : bool, optional
        True if the galaxy is a non-detection. The default is False.
    spec_res : float, optional
        The spectral resolution of the cube in km/s. The possible options are
        10 and 30. The default is 10.

    Returns
    -------
    spectrum : NumPy array
        1D array containing the y-values of the spectrum in K.
    spectrum_mJy : NumPy array
        1D array containing the y-values of the spectrum in mJy.
    spectrum_velocities : NumPy array
        1D array containing the x-values of the spectrum in km/s.

    """

    # Calculate the velocity array corresponding to the cube spectral axis in km/s
    vel_array_full, _, _ = create_vel_array(galaxy, cube, spec_res=spec_res)

    # Extract the synthesised beam major and minor axis lenghts from the cube
    # header and convert to arcseconds
    bmaj = cube.header["BMAJ"] * 3600
    bmin = cube.header["BMIN"] * 3600

    # Calculate the area of the synthesised beam in arcseconds
    if spec_res == 10:
        beam_area = calc_beam_area(bmaj, bmin, cellsize=0.1)
    elif spec_res == 30:
        beam_area = calc_beam_area(bmaj, bmin, cellsize=0.5)

    # If no CO is detected in the galaxy, take the spectrum within r50
    # (from optical observations). If r50 is not available, print the name of the
    # galaxy.
    if non_det:
        table = fits.open(glob_cat)[1]
        R50 = table.data["R50_ARCSEC"][
            table.data["KGAS_ID"] == int(galaxy.split("KGAS")[1])
        ][0]
        rad_pix = R50 / (cube.header["CDELT2"] * 3600)

        try:
            # Note that this assumes that the 0th axis of the cube is the spectral one
            aper = CircularAperture(
                [int(cube.shape[1] / 2), int(cube.shape[2] / 2)], rad_pix
            )
        except:
            print("r50 not available for " + galaxy + ".")
            return

        # The three lines below plot a 2D image of the cube (spectral axis collapsed)
        # with the aperture around r50 overlaid. This is for testing purposes only,
        # and currently there is no keyword to switch this on/off.
        # plt.figure()
        # plt.imshow(np.sum(cube.data, axis=0))
        # aper.plot(color='red', lw=2)

        # Turn the aperture into a mask to be applied to the cube
        aper_mask = aper.to_mask(method="center")
        mask = aper_mask.to_image(shape=cube.shape[1:])

        # Turn the 2D mask into a 3D one and apply it to the cube
        mask[mask == 0] = np.nan
        mask3d = np.tile(mask, (cube.shape[0], 1, 1))
        masked_data = mask3d * cube.data

        # Take the spectrum within r50
        spectrum = np.nanmean(masked_data, axis=(1, 2))

        # Calculate the spectrum in mJy
        masked_data_mJyb = brightness_temp_to_flux_dens(masked_data, bmaj, bmin)
        spectrum_mJy = np.nansum(masked_data_mJyb, axis=(1, 2)) / beam_area

    # If CO is detected in the galaxy, extract the spectrum from within the clip mask.
    # The clip mask is collapsed along the spectral axis and the full resulting 2D
    # area is used in each channel.
    else:
        # Load the clip mask, collapse it, and (re)turn it into a binary mask
        clip_mask = fits.open(
            path
            + "by_galaxy/"
            + galaxy
            + "/"
            + str(spec_res)
            + "kms/"
            + galaxy
            + "_mask_cube.fits"
        )[0]
        mask = np.sum(clip_mask.data, axis=0)
        mask = mask.astype(float)
        mask[mask > 0] = 1

        # Turn the 2D mask back into a 3D mask the shape of the cube
        mask3d = np.tile(mask, (cube.shape[0], 1, 1))
        masked_data = mask3d * cube.data
        masked_data[masked_data == 0] = np.nan

        # Take the specrum within the mask
        spectrum = np.nanmean(masked_data, axis=(1, 2))

        # Calculate the spectrum in mJy
        masked_data_mJyb = brightness_temp_to_flux_dens(masked_data, bmaj, bmin)
        spectrum_mJy = np.nansum(masked_data_mJyb, axis=(1, 2)) / beam_area

    # Add a few extra (empty) channels on either side of the spectrum, which is
    # extracted only from channels containing emission. If a boundary of the cube
    # is hit, only add as many extra channels as possible.
    if start - extra_chans < 0:
        spectrum_velocities = vel_array_full[0 : stop + extra_chans]
        spectrum = spectrum[0 : stop + extra_chans]
        spectrum_mJy = spectrum_mJy[0 : stop + extra_chans]
        # spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset
    elif stop + extra_chans > len(vel_array_full):
        spectrum_velocities = vel_array_full[start:]
        spectrum = spectrum[start:]
        spectrum_mJy = spectrum_mJy[start:]
    else:
        spectrum_velocities = vel_array_full[start - extra_chans : stop + extra_chans]
        spectrum = spectrum[start - extra_chans : stop + extra_chans]
        spectrum_mJy = spectrum_mJy[start - extra_chans : stop + extra_chans]
        # spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset

    # I'm leaving in the below two lines in case we ever want to calculate
    # the spectrum in frequency
    # rest_freq = 230538000000
    # spectrum_frequencies = rest_frequency * (1 - spectrum_velocities / 299792.458) / 1e9

    # Save the spectrum as a csv file containing both units
    csv_header = "Spectrum (K), Spectrum (mJy), Velocity (km/s)"

    if spec_res == 10:
        np.savetxt(
            path + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_spectrum.csv",
            np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
            delimiter=",",
            header=csv_header,
        )
        np.savetxt(
            path + "by_product/spectrum/10kms/" + galaxy + "_spectrum.csv",
            np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
            delimiter=",",
            header=csv_header,
        )

    elif spec_res == 30:
        np.savetxt(
            path + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_spectrum.csv",
            np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
            delimiter=",",
            header=csv_header,
        )
        np.savetxt(
            path + "by_product/spectrum/30kms/" + galaxy + "_spectrum.csv",
            np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
            delimiter=",",
            header=csv_header,
        )

    return spectrum, spectrum_mJy, spectrum_velocities


def plot_spectrum(
    galaxy,
    spectrum,
    spectrum_mJy,
    velocity,
    x_axis="velocity",
    savepath=None,
    spec_res=10,
):
    """
    Plot a spectrum and save to png and pdf.

    Parameters
    ----------
    galaxy : string
        Name of the galaxy for which the spectrum is plotted.
    spectrum : array-like
        1D array containing the y-values of the spectrum in K.
    spectrum_mJy : array-like
        1D array containing the y-values of the spectrum in mJy.
    velocity : array-like
        1D array containing the x-values of the spectrum in km/s.
    x_axis : string, optional
        Units of the x-axis. Currently 'velocity' is the only option.
        The default is 'velocity'.
    savepath : string, optional
        Path to where the plots are saved. The plots are only saved if this
        keyword is provided. The default is None.
    spec_res : float, optional
        The spectral resolution of the spectrum in km/s. The options are 10 or
        30. The default is 10.

    Raises
    ------
    AttributeError
        Raised if units for the x-axis are requested which do not exist. Currently
        commented out.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(7, 7))

    # Currently the spectrum is only saved with a velocity axis, but I am leaving
    # the options in the block comment below in case that changes.
    if x_axis == "velocity":
        ax.plot(velocity, spectrum, color="k", drawstyle="steps")
        ax.set_xlim(velocity[0], velocity[len(velocity) - 1])
        ax.set_xlabel(r"Velocity [km s$^{-1}$]")
    """
    elif x_axis == 'vel_offset':
        ax.plot(v_off, spectrum, color='k', drawstyle='steps')
        ax.set_xlim(v_off[len(v_off) - 1], v_off[0])
        ax.set_xlabel(r'Velocity offset [km s$^{-1}$]')

    elif x_axis == 'frequency':
        ax.plot(frequency, spectrum, color='k', drawstyle='steps')
        ax.set_xlim(frequency[len(frequency) - 1], frequency[0])
        ax.set_xlabel(r'Frequency [GHz]')
    else:
        raise AttributeError('Please choose between "velocity" , "vel_offset", and "frequency" for "x-axis"')
    """

    # Option to add a second axis in mJy.
    # NOTE THAT THIS CURRENTLY IS NOT WORKING AND NEEDS FIXING
    ax2 = ax.twinx()
    ax2.plot(velocity, spectrum_mJy, color="k", drawstyle="steps")
    ax2.set_ylabel("Flux Density (mJy)", color="k")

    # Plot a horizontal line through zero
    zeroline = np.zeros(len(velocity))
    ax.plot(velocity, zeroline, linestyle=":", c="k", linewidth=1)

    ax.set_ylabel("Brightness temperature [K]")

    plt.tight_layout()

    # Save the plot in png and pdf format
    if savepath:
        if spec_res == 10:
            plt.savefig(
                savepath + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_spectrum.png",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_galaxy/" + galaxy + "/10kms/" + galaxy + "_spectrum.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_product/spectrum/10kms/" + galaxy + "_spectrum.png",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_product/spectrum/10kms/" + galaxy + "_spectrum.pdf",
                bbox_inches="tight",
            )
        elif spec_res == 30:
            plt.savefig(
                savepath + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_spectrum.png",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_galaxy/" + galaxy + "/30kms/" + galaxy + "_spectrum.pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_product/spectrum/30kms/" + galaxy + "_spectrum.png",
                bbox_inches="tight",
            )
            plt.savefig(
                savepath + "by_product/spectrum/30kms/" + galaxy + "_spectrum.pdf",
                bbox_inches="tight",
            )
            
    plt.close(fig)


def get_all_spectra(
    read_path, save_path, targets, target_id, detected, chans2do, glob_cat, spec_res=10
):
    """
    Create and plot a spectrum for each of the targets provided.

    Parameters
    ----------
    read_path : string
        Directory in which the cubes are stored.
    save_path : string
        Directory in which the spectrum + plots will be saved.
    targets : list
        Contains the names of the galaxies for which a spectrum is requested.
    target_id : NumPy array
        Contains the target IDs of the galaxies (i.e. the X in KGASX).
    detected : list of bools
        Indicates for each of the galaxies whether or not it is detected in CO.
    chans2do : string
        Filename of the table containing the velocities corresponding to the
        first and last channels containing the CO line.
    glob_cat : string
        Filename of the tabel containing global parameters for the galaxy.
    spec_res : float, optional
        Spectral resolution of the cube in km/s. The options are 10 and 30.
        The default is 10.

    Returns
    -------
    None.

    """

    # Create a list of all data cube file names
    files = glob(read_path + "**/*co2-1*image.pbcor*.fits")

    # Extract the galaxy names from this list
    # NOTE: WILL HAVE TO ADJUST IF CUBE FILENAMES OR PATH CHANGE
    galaxies = list(set([f.split("/")[8].split("_")[0] for f in files]))

    # Read in the velocities corresponding to the first and last channels
    # containing line emission and store them in a dictionary
    clipping_table = pd.read_csv(chans2do)
    KGAS_ID = np.array(clipping_table["KGAS_ID"])
    minchan_v = np.array(clipping_table["minchan_v"])
    maxchan_v = np.array(clipping_table["maxchan_v"])
    clipping_vels = {
        "KGAS" + id.astype(str): [min, max]
        for id, min, max in zip(KGAS_ID, minchan_v, maxchan_v)
    }

    for galaxy in galaxies:
        # If the galaxy is not among the targets, skip it
        if not galaxy in targets:
            continue

        print("Creating spectrum for " + galaxy + ". \n")

        # Check if the galaxy is detected
        non_det = ~detected[target_id == int(galaxy.split("KGAS")[1])]

        # Create the directories in which to store the plots if they do not yet
        # exist
        if spec_res == 10:
            if not os.path.exists(save_path + "by_galaxy/" + galaxy):
                os.makedirs(save_path + "by_galaxy/" + galaxy, exist_ok=True)
            if not os.path.exists(save_path + "by_product/spectrum"):
                os.makedirs(save_path + "by_product/spectrum", exist_ok=True)
            if not os.path.exists(save_path + "by_galaxy/" + galaxy + "/10kms"):
                os.makedirs(save_path + "by_galaxy/" + galaxy + "/10kms", exist_ok=True)
            if not os.path.exists(save_path + "by_product/spectrum/10kms"):
                os.makedirs(save_path + "by_product/spectrum/10kms", exist_ok=True)
        elif spec_res == 30:
            if not os.path.exists(save_path + "by_galaxy/" + galaxy):
                os.makedirs(save_path + "by_galaxy/" + galaxy, exist_ok=True)
            if not os.path.exists(save_path + "by_product/spectrum"):
                os.makedirs(save_path + "by_product/spectrum", exist_ok=True)
            if not os.path.exists(save_path + "by_galaxy/" + galaxy + "/30kms"):
                os.makedirs(save_path + "by_galaxy/" + galaxy + "/30kms", exist_ok=True)
            if not os.path.exists(save_path + "by_product/spectrum/30kms"):
                os.makedirs(save_path + "by_product/spectrum/30kms", exist_ok=True)

        # Read in and open the data cube
        if spec_res == 10:
            cube = glob(read_path + galaxy + "/*co2-1_10.0kmps*image.pbcor*.fits")[0]
        elif spec_res == 30:
            cube = glob(read_path + galaxy + "/*co2-1_30.0kmps*image.pbcor*.fits")[0]

        cube_fits = fits.open(cube)[0]

        # Extract the minimum and maximum velocities corresponding to the emission line
        start_v = clipping_vels[galaxy][0]
        stop_v = clipping_vels[galaxy][1]

        # Create the corresponding velocity array
        vel_array, _, _ = create_vel_array(galaxy, cube_fits, spec_res=spec_res)
        start = np.argmin(abs(vel_array - start_v))
        stop = np.argmin(abs(vel_array - stop_v))

        # Create and plot the spectrum. Wrappted in a try + except because for
        # some galaxies a cube is not available in which case the script fails.
        try:
            spec, spec_mJy, vel = make_spectrum(
                cube_fits,
                galaxy,
                start,
                stop,
                save_path,
                glob_cat=glob_cat,
                extra_chans=10,
                non_det=non_det,
                spec_res=spec_res,
            )
            plot_spectrum(
                galaxy, spec, spec_mJy, vel, savepath=save_path, spec_res=spec_res
            )
        except:
            pass

    print("Done.")


if __name__ == "__main__":
    spec_res = 10
    ifu_matched = True
    version = 2.0
    overwrite = True
    print (version)
    print (spec_res)
    print (ifu_matched)
    
    if ifu_matched:
        read_path = "/arc/projects/KILOGAS/cubes/v1.0/matched/"
        save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
        save_path = "/arc/home/rock211/test_products/" + str(version) + "/matched/"
    else:
        read_path = "/arc/projects/KILOGAS/cubes/v1.0/nyquist/"
        save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"
        save_path = "/arc/home/rock211/test_products/" + str(version) + "/original/"

    chans2do = "KGAS_chans2do_v_optical_Sept25.csv"
    glob_cat = "KILOGAS_global_catalog_FWHM.fits"

    target_id = np.genfromtxt(chans2do, delimiter=",", skip_header=1, usecols=[0], dtype=int)
    if spec_res == 10:
        detected = np.genfromtxt(chans2do, delimiter=",", skip_header=1, usecols=[6], dtype=bool)
    elif spec_res == 30:
        detected = np.genfromtxt(chans2do, delimiter=",", skip_header=1, usecols=[7], dtype=bool)

    targets = ["KGAS" + str(t) for t in target_id]

    # Custom override: specific galaxies
    # targets = ['KGAS10', 'KGAS105', 'KGAS108']

    get_all_spectra(
        read_path=read_path,
        save_path=save_path,
        targets=targets,
        target_id=target_id,
        detected=detected,
        chans2do=chans2do,
        glob_cat=glob_cat,
        spec_res=spec_res,
    )