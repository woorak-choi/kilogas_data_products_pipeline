from matplotlib import pyplot as plt
import aplpy as apl
from astropy.io import fits
import numpy as np
import config_figs
from sauron_colormap import register_sauron_colormap

try:
    register_sauron_colormap()
except Exception as e:
    print(f"Colormap registration failed: {e}") # 에러 메시지를 출력하도록 수정
    
from glob import glob
from matplotlib.ticker import FuncFormatter
import pandas as pd
import os
import shutil

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False 
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # ← 추가


def moment_zero(
    mom0, galaxy, path, spec_res=10, savename=None, units="K km/s", peak=False
):
    """
    Create a moment zero image from the FITS file and save to png and pdf.

    Parameters
    ----------
    mom0 : HDU list.
        Contains the FITS image of the moment zero map and corresponding header.
    galaxy : string
        Name of the galaxy to be imaged.
    path : string
        Path to the general directory in which the images should be saved.
    spec_res : float, optional
        Spectral resolution of the cube for which the moment map is being imaged.
        Options are 10 or 30. The default is 10.
    savename : string, optional
        Name of the image file to be saved. The default is None.
    units : string, optional
        Units of the moment zero map. The default is 'K km/s'. NOTE THAT THIS
        SHOULD JUST BE READ FROM THE HEADER INSTEAD.
    peak : bool, optional
        Whether or not to make a peak temperature map instead of a moment zero map. The default is False.

    Raises
    ------
    AttributeError
        Raised if units are requested which are not possible.

    Returns
    -------
    None

    """

    fig = plt.figure(figsize=(11, 8))
    f = apl.FITSFigure(mom0, figure=fig)

    # Add the galaxy name in the upper right corner
    f.add_label(0.8, 0.9, galaxy, relative=True, size=30)

    # Overlay filled contours on the image to make it look smooth
    f.show_contour(
        mom0,
        cmap="magma_r",
        levels=np.linspace(np.nanmax(mom0.data) * 1e-9, np.nanmax(mom0.data), 20),
        filled=True,
        overlap=True,
    )

    # Adjust and ticks
    f.ticks.set_color("black")
    f.ticks.set_length(10)
    f.ticks.set_linewidth(2)
    f.tick_labels.set_xformat("hh:mm:ss")
    f.tick_labels.set_yformat("dd:mm:ss")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    f.ticks.set_minor_frequency(5)

    # Add a colourbar
    colors = plt.contourf(
        [[0, 0], [0, 0]],
        levels=np.linspace(0, np.nanmax(mom0.data) + np.nanmax(mom0.data) * 0.05, 20),
        cmap="magma_r",
    )

    # Create tick labels at intuitive intervals
    if units == "K km/s" or units == "Msol pc-2" or peak:
        if np.nanmax(mom0.data) < 0.1:
            ticks = np.arange(0, np.nanmax(mom0.data) + 0.015, 0.005)
        elif np.nanmax(mom0.data) < 0.5:
            ticks = np.arange(0, np.nanmax(mom0.data) + 0, 0.05)
        elif np.nanmax(mom0.data) < 1:
            ticks = np.arange(0, np.nanmax(mom0.data) + 0.1, 0.1)
        elif np.nanmax(mom0.data) < 2:
            ticks = np.arange(0, np.nanmax(mom0.data) + 0.2, 0.2)
        elif np.nanmax(mom0.data) < 5:
            ticks = np.arange(0, np.nanmax(mom0.data) + 0.5, 0.5)
        elif np.nanmax(mom0.data) < 10:
            ticks = np.arange(0, np.nanmax(mom0.data) + 1, 1)
        elif np.nanmax(mom0.data) < 20:
            ticks = np.arange(0, np.nanmax(mom0.data) + 1, 2)
        elif np.nanmax(mom0.data) < 100:
            ticks = np.arange(0, np.nanmax(mom0.data) + 5, 10)
        elif np.nanmax(mom0.data) < 200:
            ticks = np.arange(0, np.nanmax(mom0.data) + 10, 20)
        elif np.nanmax(mom0.data) < 1000:
            ticks = np.arange(0, np.nanmax(mom0.data) + 20, 40)
        else:
            ticks = np.arange(0, np.nanmax(mom0.data) + 100, 200)

        cbar = fig.colorbar(colors, ticks=ticks)

    # In case of these units the numbers are large, so need to be treated
    # differently
    elif units == "K km/s pc^2" or units == "Msol/pix":
        # Extract the base and exponent of the maximum in the moment 0 map
        # if the number was written in scientific format
        exponent = np.floor(np.log10(np.nanmax(mom0.data)))
        base = np.nanmax(mom0.data) / (10**exponent)

        # Create intuitive locations for the tick marks depending on the max.
        # base number
        if base < 0.1:
            ticks = np.arange(0, base, 0.005)
        elif base < 0.5:
            ticks = np.arange(0, base, 0.05)
        elif base < 1:
            ticks = np.arange(0, base, 0.1)
        elif base < 2:
            ticks = np.arange(0, base, 0.2)
        elif base < 5:
            ticks = np.arange(0, base, 0.5)
        else:
            ticks = np.arange(0, base, 1)

        ticks *= 10**exponent

        # Mini function to format the ticks. If you simply want the "x10^exp"
        # at the top of the colour bar per the default, comment out the line
        # that calls this function
        def tick_formatter(x, pos):
            exponent = np.floor(np.log10(np.nanmax(mom0.data)))
            base = x / (10**exponent)

            return f"{base:.1f}$\\times 10^{int(exponent)}$"

        cbar = fig.colorbar(colors, ticks=ticks)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))

    # Set the label for the colour bar
    if peak:
        cbar.set_label("Peak temperature [K]")
    elif units == "K km/s pc^2":
        if "err" in savename:
            cbar.set_label(r"CO luminosity error [K km s$^{-1}$ pc$^2$]")
        else:
            cbar.set_label(r"CO luminosity [K km s$^{-1}$ pc$^2$]")
    elif units == "K km/s":
        if "err" in savename:
            cbar.set_label(r"CO surface brightness error [K km s$^{-1}$]")
        else:
            cbar.set_label(r"CO surface brightness [K km s$^{-1}$]")
    elif units == "Msol pc-2":
        cbar.set_label(r"mol. mass surface density [M$_\odot$ pc$^{-2}$]")
    elif units == "Msol/pix":
        cbar.set_label(r"mol. mass [M$_\odot$]")
    else:
        raise AttributeError(
            'Please choose from "K km/s pc^2", "K km/s", \
                             "Msol pc-2", and "M_Sun/pc^2", or set "peak" to True'
        )

    # Show the synthesised beam of the observations
    f.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
    f.beam.set_edgecolor("k")
    f.beam.set_facecolor("None")
    f.beam.set_borderpad(1)

    # Show a scalebar
    # NOTE: This is not currently set up as distances are/were not known for each
    # galaxy. Can be implemented if so desired.
    # if self.galaxy.distance:
    #    length = np.degrees(1e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
    #    fig.add_scalebar(length=length, label='1 kpc', frame=False)
    #    fig.scalebar.set_linewidth(5)

    plt.tight_layout()

    if savename:
        # ── [Optimization 1: Render once, then copy to save time] ──
        primary_png = path + savename + ".png"
        primary_pdf = path + savename + ".pdf"
        
        secondary_dir = path.split("by_galaxy")[0] + "by_product/moment_maps/" + str(spec_res) + "kms/"
        secondary_png = secondary_dir + savename + ".png"
        secondary_pdf = secondary_dir + savename + ".pdf"

        # Save (render) only ONCE
        plt.savefig(primary_png, bbox_inches="tight")
        plt.savefig(primary_pdf, bbox_inches="tight")

        # Copy the rendered files to the secondary location (Extremely fast disk I/O)
        os.makedirs(secondary_dir, exist_ok=True)
        shutil.copy(primary_png, secondary_png)
        shutil.copy(primary_pdf, secondary_pdf)
        
    plt.close(fig)


def moment_1_2(mom, galaxy, moment, path, spec_res=10, savename=None, clipping_vels=None):
    """
    Creates images for the moment one and two maps and saves them to png and pdf.

    Parameters
    ----------
    mom : HDU list
        Contains the FITS image and corresponding header.
    galaxy : string
        The galaxy that is being imaged.
    moment : float
        Which moment map is being imaged. Options are 1 or 2.
    path : string
        Path to where the images are going to be saved.
    spec_res : float, optional
        Spectral resolution of the cube for which the moment map is imaged. Options
        are 10 and 30 km/s. The default is 10. NOTE: this should just be read from the header.
    savename : string, optional
        Filename of the image to be saved. The image is only saved if this is provided. The default is None.
    chans2do : string, optional
        Path to the table containing the numbers of the channels that contain emission. The default is None.

    Returns
    -------
    None.

    """
    # Load the velocity array
    vel_array = np.load(
        path
        + "by_galaxy/"
        + galaxy
        + "/"
        + str(spec_res)
        + "kms/"
        + savename.split("_mom1")[0].split("_mom2")[0]
        + "_vel_array.npy"
    )

    # Round the systemic velocity in case you want to show it in the figure
    # sysvel = (sysvel + 5) // 10 * 10

    fig = plt.figure(figsize=(11, 8))

    # Show the image in colour
    f = apl.FITSFigure(mom, figure=fig)

    # Format tick labels
    f.ticks.set_color("black")
    f.ticks.set_length(10)
    f.ticks.set_linewidth(2)
    f.tick_labels.set_xformat("hh:mm:ss")
    f.tick_labels.set_yformat("dd:mm:ss")
    f.ticks.show()
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    f.ticks.set_minor_frequency(5)

    # Add a colourbar
    if moment == 2:
        vrange2 = np.nanmax(mom.data[np.isfinite(mom.data)])

        # If the velocity range in the moment two map is very large (unphysical),
        # simply set it to 100. NOTE: can make this more clever.
        if vrange2 > 100:
            vrange2 = 100

        f.show_contour(
            mom,
            cmap="sauron",
            levels=np.linspace(0, vrange2, len(vel_array)),
            vmin=0,
            vmax=vrange2,
            extend="both",
            filled=True,
            overlap=True,
        )

        colors = plt.contourf(
            [[0, 0], [0, 0]],
            levels=np.linspace(0, vrange2, len(vel_array)),
            cmap="sauron",
        )

        # Set the ticks in an intuitive way depending on the maximum velocity
        # dispersion
        if vrange2 < 11:
            ticks = np.arange(0, vrange2 + 1, 1)
        elif vrange2 < 100:
            ticks = np.arange(0, vrange2 + 10, 10)
        elif vrange2 < 500:
            ticks = np.arange(0, vrange2 + 20, 20)
        else:
            ticks = []

        cbar = fig.colorbar(colors, ticks=ticks)
        if "err" in savename:
            cbar.set_label(r"Observed $\sigma_v$ error [km s$^{-1}$]")
        else:
            cbar.set_label(r"Observed $\sigma_v$ [km s$^{-1}$]")

    elif moment == 1:
        # ── [Optimization 2: Use pre-loaded dictionary instead of reading CSV] ──
        if mom.header["BTYPE"] == "co_vel" and clipping_vels is not None:
            vmin = clipping_vels[galaxy][0]
            vmax = clipping_vels[galaxy][1]
        else:
            vmin = 0
            vmax = 30

        f.show_contour(
            mom,
            cmap="sauron",
            levels=np.linspace(vmin, vmax, len(vel_array)),
            vmin=vmin,
            vmax=vmax,
            extend="both",
            filled=True,
            overlap=True,
        )

        colors = plt.contourf(
            [[0, 0], [0, 0]],
            levels=np.linspace(vmin, vmax, len(vel_array)),
            cmap="sauron",
        )

        # Calculate the velocity range of the moment one map
        vrange = vel_array[-1] - vel_array[0]

        if vrange < 16:
            tickarr = np.arange(round(vmin), round(vmax), 3)
        elif vrange < 60:
            tickarr = np.arange(round(vmin / 5) * 5 - 5, round(vmax / 5) * 5 + 5, 10)
        elif vrange < 130:
            tickarr = np.arange(
                round(vmin / 10) * 10 - 10, round(vmax / 10) * 10 + 10, 20
            )
        elif vrange < 1000:
            ticks = np.arange(
                round(vmin / 10) * 10 - 10, round(vmax / 10) * 10 + 10, 50
            )
        else:
            ticks = []

        try:
            cbar = fig.colorbar(colors, ticks=ticks)
        except:
            ticks = np.concatenate((tickarr, [0], abs(tickarr)))
            cbar = fig.colorbar(colors, ticks=ticks)
        if "err" in savename:
            cbar.set_label(r"Velocity error [km s$^{-1}$]")
        else:
            cbar.set_label(r"Velocity [km s$^{-1}$]")

    # show the beam of the observations
    f.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
    f.beam.set_edgecolor("k")
    f.beam.set_facecolor("None")
    f.beam.set_borderpad(1)

    # show a scalebar
    # if self.galaxy.distance:
    #    length = np.degrees(1.e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
    #    fig.add_scalebar(length=length, label='1 kpc', frame=False)
    #    fig.scalebar.set_linewidth(5)

    plt.tight_layout()

    if savename:
        # ── [Optimization 1: Render once, then copy to save time] ──
        primary_png = path + "by_galaxy/" + galaxy + "/" + str(spec_res) + "kms/" + savename + ".png"
        primary_pdf = path + "by_galaxy/" + galaxy + "/" + str(spec_res) + "kms/" + savename + ".pdf"
        
        secondary_dir = path + "by_product/moment_maps/" + str(spec_res) + "kms/"
        secondary_png = secondary_dir + savename + ".png"
        secondary_pdf = secondary_dir + savename + ".pdf"

        # Save (render) only ONCE
        plt.savefig(primary_png, bbox_inches="tight")
        plt.savefig(primary_pdf, bbox_inches="tight")

        # Copy to the secondary location
        os.makedirs(secondary_dir, exist_ok=True)
        shutil.copy(primary_png, secondary_png)
        shutil.copy(primary_pdf, secondary_pdf)
        
    plt.close(fig)

def perform_moment_imaging(glob_path, targets, chans2do, spec_res=10):
    """
    Read in the moment map FITS files and create the corresponding png/pdf
    images for each.

    Parameters
    ----------
    glob_path : string
        Path to where the products are stored.
    targets : list
        Contains the galaxy names of the targets to be imaged.
    chans2do : string
        Path to the table containing the velocities.
    spec_res : float, optional
        The spectral resolution of the original cube in km/s. Can be either
        10 or 30. The default is 10.

    Returns
    -------
    None.

    """

    # Find the relevant files and extract the corresponding galaxy name.
    # NOTE: this will break if the path to the products changes and needs to be
    # improved.
    # ── [Optimization 2: Read CSV ONCE before the loop to prevent redundant I/O] ──
    clipping_vels = None
    if chans2do and os.path.exists(chans2do):
        clipping_table = pd.read_csv(chans2do)
        KGAS_ID = np.array(clipping_table["KGAS_ID"])
        minchan_v = np.array(clipping_table["minchan_v"])
        maxchan_v = np.array(clipping_table["maxchan_v"])
        clipping_vels = {
            "KGAS" + str(id): [min_v, max_v]
            for id, min_v, max_v in zip(KGAS_ID, minchan_v, maxchan_v)
        }

    # ── [Bug Fix: Safely extract galaxy names regardless of path depth] ──
    files = glob(glob_path + "by_galaxy/*/")
    galaxies = list(set([os.path.basename(f.rstrip("/")) for f in files]))

    # Loop over the list of galaxies and create the moment map images.
    for galaxy in galaxies:
        if not galaxy in targets:
            continue
        else:
            print(galaxy)

        path = glob_path + "by_galaxy/" + galaxy + "/" + str(spec_res) + "kms/"
        # I sometimes switch this on if the code gets stuck so it doesn't start
        # remaking all the duplicate images.
        # if os.path.exists(path + galaxy + '_Ico_K_kms-1.png'):
        #    continue

        # Read in the different types of moment maps
        mom0_K_kmss = glob(path + "*Ico*.fits")
        mom0_K_kms_pc2s = glob(path + "*Lco*.fits")
        mom0_Msol_pc2 = glob(path + "*mmol_pc-2*.fits")
        mom0_Msol_pix = glob(path + "*mmol_pix-1*.fits")
        peakTs = glob(path + "*peak_temp_k*.fits")
        mom1s = glob(path + "*mom1*.fits")
        mom2s = glob(path + "*mom2*.fits")

        # Create the images for the moment maps
        # NOTE: This is wrapped in a try + except I think because the moment 1 or 2
        # maps failed for some galaxies. It should be checked if this is still
        # necessary and until then commented out while debugging.
        try:
            for mom0 in mom0_K_kmss:
                moment_zero(
                    fits.open(mom0)[0],
                    galaxy=galaxy,
                    path=path,
                    savename=mom0.split("/")[-1].split(".fits")[0],
                    spec_res=spec_res,
                    units="K km/s",
                    peak=False,
                )
            for mom0 in mom0_K_kms_pc2s:
                moment_zero(
                    fits.open(mom0)[0],
                    galaxy=galaxy,
                    path=path,
                    savename=mom0.split("/")[-1].split(".fits")[0],
                    spec_res=spec_res,
                    units="K km/s pc^2",
                    peak=False,
                )
            for mom0 in mom0_Msol_pc2:
                moment_zero(
                    fits.open(mom0)[0],
                    galaxy=galaxy,
                    path=path,
                    savename=mom0.split("/")[-1].split(".fits")[0],
                    spec_res=spec_res,
                    units="Msol pc-2",
                    peak=False,
                )
            for mom0 in mom0_Msol_pix:
                moment_zero(
                    fits.open(mom0)[0],
                    galaxy=galaxy,
                    path=path,
                    savename=mom0.split("/")[-1].split(".fits")[0],
                    spec_res=spec_res,
                    units="Msol/pix",
                    peak=False,
                )
            for peakT in peakTs:
                moment_zero(
                    fits.open(peakT)[0],
                    galaxy=galaxy,
                    path=path,
                    savename=peakT.split("/")[-1].split(".fits")[0],
                    spec_res=spec_res,
                    peak=True,
                )
            for mom1 in mom1s:
                moment_1_2(
                    fits.open(mom1)[0],
                    savename=mom1.split("/")[-1].split(".fits")[0],
                    galaxy=galaxy,
                    moment=1,
                    path=glob_path,
                    spec_res=spec_res,
                    clipping_vels=clipping_vels, # <--- 딕셔너리를 직접 전달!
                )
            for mom2 in mom2s:
                moment_1_2(
                    fits.open(mom2)[0],
                    savename=mom2.split("/")[-1].split(".fits")[0],
                    galaxy=galaxy,
                    moment=2,
                    path=glob_path,
                    spec_res=spec_res,
                )
        except Exception as e:
            print(f"Error processing {galaxy}: {e}")


if __name__ == "__main__":
    # NOTE: all hardcoded for now as it isn't really used, but can improve
    # to take user input.

    version = 3.0
    ifu_matched = True
    spec_res = 30

    if ifu_matched:
        glob_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
        glob_path = "/arc/home/rock211/test_products/" + str(version) + "/matched/"
    else:
        glob_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"
        glob_path = "/arc/home/rock211/test_products/" + str(version) + "/original/"
        
    targets = ["KGAS107"]
    chans2do = "KGAS_chans2do_v_optical_Sept25.csv"

    perform_moment_imaging(glob_path, targets, chans2do, spec_res=spec_res)
