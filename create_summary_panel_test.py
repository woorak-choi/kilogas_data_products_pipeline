"""
create_summary_panel.py

Creates a single summary figure per galaxy containing all key data products
in a grid layout. Uses matplotlib WCSAxes directly (instead of APLpy) for speed.

For detected galaxies: 9 panels (8 maps + 1 spectrum) in a 3x3 grid
For non-detected galaxies: 2 panels (upper limit map + spectrum)

Author: Woorak Choi
Date: March 2026
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
import numpy as np
from glob import glob
import os
import pandas as pd

# Register sauron colormap; use coolwarm as fallback
try:
    from sauron_colormap import register_sauron_colormap
    register_sauron_colormap()
    VELOCITY_CMAP = 'sauron'
except:
    VELOCITY_CMAP = 'coolwarm'


def _add_beam(ax, header):
    """Add synthesized beam ellipse to the lower-left corner of the axes."""
    try:
        bmaj = header['BMAJ']  # degrees
        bmin = header['BMIN']  # degrees
        bpa = header.get('BPA', 0)  # degrees
        pixscale = abs(header['CDELT2'])  # degrees/pixel

        # Convert beam size from degrees to pixels
        bmaj_pix = bmaj / pixscale
        bmin_pix = bmin / pixscale

        # Place beam in lower-left corner (in pixel coordinates)
        corner_x = bmaj_pix * 1.5
        corner_y = bmaj_pix * 1.5

        beam = Ellipse((corner_x, corner_y), bmin_pix, bmaj_pix, angle=bpa,
                        edgecolor='black', facecolor='none', linewidth=1.5,
                        transform=ax.get_transform('pixel'))
        ax.add_patch(beam)
    except:
        pass


def _add_map_panel(fig, rect, fits_hdu, title, cmap='magma_r',
                   vmin=None, vmax=None, cbar_label=''):
    """
    Add a single map panel using WCSAxes (much faster than APLpy).

    Parameters
    ----------
    fig : matplotlib Figure
    rect : list [left, bottom, width, height] in figure coords (0-1)
    fits_hdu : astropy.io.fits PrimaryHDU
        2D FITS image.
    title : str
    cmap : str
    vmin, vmax : float, optional
    cbar_label : str
    """
    data = fits_hdu.data
    header = fits_hdu.header

    # Build 2D WCS (drop spectral axis if present)
    wcs_full = WCS(header, naxis=2)

    # Auto-determine vmin/vmax if not set
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None

    if vmin is None:
        vmin = np.nanmin(finite_data)
    if vmax is None:
        vmax = np.nanmax(finite_data)

    # Create axes with WCS projection
    ax = fig.add_axes(rect, projection=wcs_full)

    # Display image
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Title
    ax.set_title(title, fontsize=9, fontweight='bold')

    # Axis labels
    ax.set_xlabel('RA (J2000)', fontsize=7)
    ax.set_ylabel('Dec (J2000)', fontsize=7)

    # Tick formatting
    ra = ax.coords['ra']
    dec = ax.coords['dec']
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')
    ra.set_ticklabel(size=6)
    dec.set_ticklabel(size=6)
    ra.set_ticks_position('b')
    dec.set_ticks_position('l')

    # Beam
    _add_beam(ax, header)

    return ax


def _add_spectrum_panel(ax, galaxy, path, spec_res=10):
    """Add the spectrum plot to an axes panel."""
    spec_file = glob(path + 'by_galaxy/' + galaxy + '/' + str(spec_res)
                     + 'kms/' + galaxy + '_spectrum.csv')

    if not spec_file:
        ax.text(0.5, 0.5, 'No spectrum\navailable', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        ax.set_title('CO(2-1) Spectrum', fontsize=9, fontweight='bold')
        return

    data = np.genfromtxt(spec_file[0], delimiter=',', skip_header=1)
    spectrum_K = data[:, 0]
    spectrum_mJy = data[:, 1]
    velocity = data[:, 2]

    ax.plot(velocity, spectrum_K, color='k', drawstyle='steps', linewidth=1)
    ax.axhline(0, linestyle=':', color='k', linewidth=0.5)
    ax.set_xlim(velocity[0], velocity[-1])
    ax.set_xlabel(r'Velocity [km s$^{-1}$]', fontsize=7)
    ax.set_ylabel('Brightness temperature [K]', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title('CO(2-1) Spectrum', fontsize=9, fontweight='bold')

    # Second y-axis for mJy
    ax2 = ax.twinx()
    ax2.plot(velocity, spectrum_mJy, color='k', drawstyle='steps',
             linewidth=0.5, alpha=0)
    ax2.set_ylabel('Flux Density [mJy]', fontsize=7, color='gray')
    ax2.tick_params(labelsize=6, colors='gray')

    ylim_K = ax.get_ylim()
    K_range = ylim_K[1] - ylim_K[0]
    if K_range != 0:
        mJy_range = np.nanmax(spectrum_mJy) - np.nanmin(spectrum_mJy)
        ratio = mJy_range / K_range
        ax2.set_ylim(ylim_K[0] * ratio, ylim_K[1] * ratio)


def create_summary_detected(galaxy, path, chans2do, spec_res=10,
                             savepath=None, overwrite=True):
    """
    Create a 3x3 summary panel for a detected galaxy.

    Layout:
        Row 1: Ico (K km/s)  |  Sigma_mol  |  Peak Temperature
        Row 2: Mom1 (vel)    |  Mom2 (disp) |  S/N map
        Row 3: Ico error     |  Ico UL      |  Spectrum
    """
    if savepath is None:
        savepath = path

    if not overwrite:
        check_file = (savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res)
                      + 'kms/' + galaxy + '_summary.png')
        if os.path.exists(check_file):
            print(f'  Skipping summary for {galaxy} (exists, overwrite=False)')
            return

    map_dir = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'

    # ── Map configurations ──
    # (glob_pattern, title, cmap, cbar_label, vmin_override, vmax_override)
    # vmin/vmax = None means auto
    map_configs = [
        ('*Ico_K_kms-1.fits',
         r'Moment 0 ($I_{\rm CO}$)', 'magma_r',
         r'$I_{\rm CO}$ [K km s$^{-1}$]', 0, None),
        ('*mmol_pc-2.fits',
         r'$\Sigma_{\rm mol}$', 'magma_r',
         r'$\Sigma_{\rm mol}$ [M$_\odot$ pc$^{-2}$]', 0, None),
        ('*peak_temp_k.fits',
         'Peak Temperature', 'magma_r',
         r'$T_{\rm peak}$ [K]', 0, None),
        ('*mom1.fits',
         'Velocity (Mom 1)', VELOCITY_CMAP,
         r'$v$ [km s$^{-1}$]', 'mom1', 'mom1'),
        ('*mom2.fits',
         r'$\sigma_v$ (Mom 2)', VELOCITY_CMAP,
         r'$\sigma_v$ [km s$^{-1}$]', 0, 'mom2'),
        ('*mom0_SN.fits',
         'Signal-to-Noise', 'magma_r',
         'S/N', 0, None),
        ('*Ico_K_kms-1_err.fits',
         r'$I_{\rm CO}$ Error', 'magma_r',
         r'$\sigma(I_{\rm CO})$ [K km s$^{-1}$]', 0, None),
        ('*Ico_K_kms-1_ul.fits',
         r'$I_{\rm CO}$ Upper Limit', 'magma_r',
         r'UL [K km s$^{-1}$]', None, None),
    ]

    # Get velocity range for mom1 from chans2do
    clipping_table = pd.read_csv(chans2do)
    kgasid = int(galaxy.split('KGAS')[1])
    row = clipping_table[clipping_table['KGAS_ID'] == kgasid]
    vmin_mom1 = row['minchan_v'].iloc[0] if len(row) > 0 else None
    vmax_mom1 = row['maxchan_v'].iloc[0] if len(row) > 0 else None

    # ── Layout: 3x3 grid with explicit positions ──
    ncols, nrows = 3, 3
    left_margin = 0.06
    right_margin = 0.02
    bottom_margin = 0.04
    top_margin = 0.06
    hspace = 0.12
    vspace = 0.10

    panel_w = (1.0 - left_margin - right_margin - (ncols - 1) * hspace) / ncols
    panel_h = (1.0 - bottom_margin - top_margin - (nrows - 1) * vspace) / nrows

    def get_rect(r, c):
        left = left_margin + c * (panel_w + hspace)
        bottom = 1.0 - top_margin - (r + 1) * panel_h - r * vspace
        return [left, bottom, panel_w, panel_h]

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(galaxy + '  (' + str(spec_res) + ' km/s)',
                 fontsize=16, fontweight='bold', y=0.98)

    # ── Plot 8 map panels + 1 spectrum panel ──
    panel_idx = 0
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            rect = get_rect(row_idx, col_idx)

            if panel_idx < len(map_configs):
                pattern, title, cmap, cbar_label, vmin_cfg, vmax_cfg = map_configs[panel_idx]

                # Find FITS file (exclude wrong variants for base maps)
                files = glob(map_dir + pattern)
                if ('_err' not in pattern and '_ul' not in pattern
                        and '_SN' not in pattern):
                    files = [f for f in files
                             if '_err' not in f and '_ul' not in f]

                if files:
                    hdu = fits.open(files[0])[0]

                    # Resolve vmin/vmax
                    vmin_panel = vmin_cfg
                    vmax_panel = vmax_cfg

                    if vmin_cfg == 'mom1':
                        vmin_panel = vmin_mom1
                    if vmax_cfg == 'mom1':
                        vmax_panel = vmax_mom1
                    if vmax_cfg == 'mom2':
                        vmax_panel = np.nanmax(
                            hdu.data[np.isfinite(hdu.data)])
                        if vmax_panel > 100:
                            vmax_panel = 100

                    try:
                        _add_map_panel(fig, rect, hdu, title, cmap=cmap,
                                       vmin=vmin_panel, vmax=vmax_panel,
                                       cbar_label=cbar_label)
                    except Exception as e:
                        ax = fig.add_axes(rect)
                        ax.text(0.5, 0.5,
                                f'Error loading\n{pattern}\n{str(e)[:50]}',
                                transform=ax.transAxes, ha='center',
                                va='center', fontsize=8, color='red')
                        ax.set_title(title, fontsize=9, fontweight='bold')
                else:
                    ax = fig.add_axes(rect)
                    ax.text(0.5, 0.5, 'Not available',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=10, color='gray')
                    ax.set_title(title, fontsize=9, fontweight='bold')

            elif panel_idx == len(map_configs):
                # Spectrum panel
                ax = fig.add_axes(rect)
                _add_spectrum_panel(ax, galaxy, path, spec_res=spec_res)

            panel_idx += 1

    # ── Save ──
    outdir_gal = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    outdir_prod = savepath + 'by_product/summary/' + str(spec_res) + 'kms/'
    os.makedirs(outdir_gal, exist_ok=True)
    os.makedirs(outdir_prod, exist_ok=True)

    fig.savefig(outdir_gal + galaxy + '_summary_test.png', dpi=150,
                bbox_inches='tight')
    fig.savefig(outdir_gal + galaxy + '_summary_test.pdf', bbox_inches='tight')
    fig.savefig(outdir_prod + galaxy + '_summary_test.png', dpi=150,
                bbox_inches='tight')
    fig.savefig(outdir_prod + galaxy + '_summary_test.pdf', bbox_inches='tight')

    plt.close(fig)
    print(f'  Summary panel saved for {galaxy} ({spec_res} km/s)')


def create_summary_nondetected(galaxy, path, spec_res=10,
                                savepath=None, overwrite=True):
    """
    Create a 1x2 summary panel for a non-detected galaxy.
    Layout: Ico upper limit  |  Spectrum
    """
    if savepath is None:
        savepath = path

    if not overwrite:
        check_file = (savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res)
                      + 'kms/' + galaxy + '_summary.png')
        if os.path.exists(check_file):
            print(f'  Skipping summary for {galaxy} (exists, overwrite=False)')
            return

    map_dir = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'

    # Layout
    left_margin = 0.06
    right_margin = 0.02
    bottom_margin = 0.08
    top_margin = 0.10
    hspace = 0.12
    panel_w = (1.0 - left_margin - right_margin - hspace) / 2
    panel_h = 1.0 - bottom_margin - top_margin

    rect_left = [left_margin, bottom_margin, panel_w, panel_h]
    rect_right = [left_margin + panel_w + hspace, bottom_margin,
                  panel_w, panel_h]

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(galaxy + '  (' + str(spec_res) + ' km/s)  --  Non-detection',
                 fontsize=14, fontweight='bold', y=0.97)

    # Panel 1: Ico upper limit
    ul_files = glob(map_dir + '*Ico_K_kms-1_ul.fits')
    if ul_files:
        hdu = fits.open(ul_files[0])[0]
        try:
            _add_map_panel(fig, rect_left, hdu,
                           r'$I_{\rm CO}$ Upper Limit (3$\sigma$)',
                           cmap='magma_r',
                           cbar_label=r'UL [K km s$^{-1}$]')
        except Exception as e:
            ax = fig.add_axes(rect_left)
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='red')
            ax.set_title(r'$I_{\rm CO}$ Upper Limit',
                         fontsize=9, fontweight='bold')
    else:
        ax = fig.add_axes(rect_left)
        ax.text(0.5, 0.5, 'Not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        ax.set_title(r'$I_{\rm CO}$ Upper Limit',
                     fontsize=9, fontweight='bold')

    # Panel 2: Spectrum
    ax_spec = fig.add_axes(rect_right)
    _add_spectrum_panel(ax_spec, galaxy, path, spec_res=spec_res)

    # Save
    outdir_gal = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    outdir_prod = savepath + 'by_product/summary/' + str(spec_res) + 'kms/'
    os.makedirs(outdir_gal, exist_ok=True)
    os.makedirs(outdir_prod, exist_ok=True)

    fig.savefig(outdir_gal + galaxy + '_summary_test.png', dpi=150,
                bbox_inches='tight')
    fig.savefig(outdir_gal + galaxy + '_summary_test.pdf', bbox_inches='tight')
    fig.savefig(outdir_prod + galaxy + '_summary_test.png', dpi=150,
                bbox_inches='tight')
    fig.savefig(outdir_prod + galaxy + '_summary_test.pdf', bbox_inches='tight')

    plt.close(fig)
    print(f'  Summary panel saved for {galaxy} ({spec_res} km/s, non-detection)')


def perform_summary_imaging(path, detections, non_detections, chans2do,
                             spec_res=10, savepath=None, overwrite=True):
    """Create summary panels for all galaxies."""
    if savepath is None:
        savepath = path

    print(f'Creating summary panels ({spec_res} km/s)...')
    print(f'  Detected galaxies: {len(detections)}')
    print(f'  Non-detected galaxies: {len(non_detections)}')

    for galaxy in detections:
        try:
            create_summary_detected(galaxy, path, chans2do,
                                    spec_res=spec_res, savepath=savepath,
                                    overwrite=overwrite)
        except Exception as e:
            print(f'  ERROR for {galaxy}: {str(e)}')

    for galaxy in non_detections:
        try:
            create_summary_nondetected(galaxy, path, spec_res=spec_res,
                                       savepath=savepath, overwrite=overwrite)
        except Exception as e:
            print(f'  ERROR for {galaxy}: {str(e)}')

    print('Summary panel creation complete.')


if __name__ == '__main__':
    spec_res = 30
    ifu_matched = False
    version = 2.0
    overwrite = True

    print (version)
    print (spec_res)
    print (ifu_matched)
    
    if ifu_matched:
        path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
        path = "/arc/home/rock211/test_products/" + str(version) + "/matched/"
    else:
        path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"
        path = "/arc/home/rock211/test_products/" + str(version) + "/original/"

    chans2do = "KGAS_chans2do_v_optical_Sept25.csv"

    target_id = np.genfromtxt(chans2do, delimiter=',', skip_header=1,
                               usecols=[0], dtype=int)
    if spec_res == 10:
        detected = np.genfromtxt(chans2do, delimiter=',', skip_header=1,
                                  usecols=[6], dtype=bool)
    elif spec_res == 30:
        detected = np.genfromtxt(chans2do, delimiter=',', skip_header=1,
                                  usecols=[7], dtype=bool)

    detections = ['KGAS' + str(t) for t, d in zip(target_id, detected) if d]
    non_detections = ['KGAS' + str(t) for t, d in zip(target_id, detected)
                      if not d]

    # # Custom override: specific galaxies
    # targets = ['KGAS10', 'KGAS12', 'KGAS13', 'KGAS15', 'KGAS16', 'KGAS17']
    # detections = [g for g in targets if g in detections]
    # non_detections = [g for g in targets if g in non_detections]

    perform_summary_imaging(
        path=path,
        detections=detections,
        non_detections=non_detections,
        chans2do=chans2do,
        spec_res=spec_res,
        overwrite=overwrite,
    )
