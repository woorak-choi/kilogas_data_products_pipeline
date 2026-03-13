"""
create_summary_panel.py (UL Reference Zoom Version)

Creates a single summary figure per galaxy.
Updates:
- Zooms based specifically on the non-NaN extent of the Upper Limit (*_ul.fits) map.
- Maintains a consistent field of view across all 9 panels.
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
import numpy as np
from glob import glob
import os
import pandas as pd
import shutil

# Register sauron colormap; use coolwarm as fallback
try:
    from sauron_colormap import register_sauron_colormap
    register_sauron_colormap()
    VELOCITY_CMAP = 'sauron'
except:
    VELOCITY_CMAP = 'coolwarm'


def _get_data_extent(map_dir, pad=1):
    """
    UL 맵(*_ul.fits)을 기준으로 유효한 데이터가 있는 픽셀 범위를 계산합니다.
    이 범위는 모든 패널의 공통 FOV로 사용됩니다.
    """
    # 상한값(UL) 파일만 특정해서 찾습니다.
    ul_files = glob(os.path.join(map_dir, "*_ul.fits"))
    
    if not ul_files:
        # UL 파일이 없으면 줌인을 수행하지 않음
        return None
        
    try:
        # 첫 번째로 검색된 UL 파일을 엽니다.
        with fits.open(ul_files[0]) as h:
            data = h[0].data
            # NaN이 아닌 유효한 데이터 포인트의 인덱스를 찾습니다.
            finite = np.where(np.isfinite(data))
            if len(finite[0]) > 0:
                y_min, y_max = np.min(finite[0]), np.max(finite[0])
                x_min, x_max = np.min(finite[1]), np.max(finite[1])
                
                # 계산된 범위에 여백(pad)을 더해 반환합니다.
                return [x_min - pad, x_max + pad, y_min - pad, y_max + pad]
    except Exception as e:
        print(f"  Warning: Could not calculate extent from UL map for {map_dir}: {e}")
        return None
    
    return None


def _add_beam(ax, header):
    """Add synthesized beam ellipse relative to the visible area (lower-left)."""
    try:
        bmaj, bmin = header['BMAJ'], header['BMIN']
        bpa = header.get('BPA', 0)
        pixscale = abs(header['CDELT2'])

        bmaj_pix = bmaj / pixscale
        bmin_pix = bmin / pixscale

        # 현재 줌이 적용된 화면 영역을 기준으로 빔 위치 결정
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        corner_x = xlim[0] + (xlim[1] - xlim[0]) * 0.1
        corner_y = ylim[0] + (ylim[1] - ylim[0]) * 0.1

        beam = Ellipse((corner_x, corner_y), bmin_pix, bmaj_pix, angle=bpa,
                        edgecolor='black', facecolor='none', linewidth=1.5,
                        transform=ax.get_transform('pixel'))
        ax.add_patch(beam)
    except:
        pass


def _add_map_panel(fig, rect, fits_hdu, title, cmap='magma_r',
                   vmin=None, vmax=None, cbar_label='', extent=None):
    """Add a single map panel with optional cropping (extent)."""
    data = fits_hdu.data
    header = fits_hdu.header
    wcs_full = WCS(header, naxis=2)

    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None

    if vmin is None: vmin = np.nanmin(finite_data)
    if vmax is None: vmax = np.nanmax(finite_data)

    ax = fig.add_axes(rect, projection=wcs_full)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')

    # UL 맵 기준 범위 적용
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)

    ra, dec = ax.coords['ra'], ax.coords['dec']
    ra.set_major_formatter('hh:mm:ss'); dec.set_major_formatter('dd:mm:ss')
    ra.set_ticklabel(size=6); dec.set_ticklabel(size=6)

    _add_beam(ax, header)
    return ax


def _add_spectrum_panel(ax, galaxy, path, spec_res=10):
    spec_file = glob(path + 'by_galaxy/' + galaxy + '/' + str(spec_res)
                     + 'kms/' + galaxy + '_spectrum.csv')

    if not spec_file:
        ax.text(0.5, 0.5, 'No spectrum available', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
        return

    data = np.genfromtxt(spec_file[0], delimiter=',', skip_header=1)
    spectrum_K, spectrum_mJy, velocity = data[:, 0], data[:, 1], data[:, 2]

    ax.plot(velocity, spectrum_K, color='k', drawstyle='steps', linewidth=1)
    ax.axhline(0, linestyle=':', color='k', linewidth=0.5)
    ax.set_xlim(velocity[0], velocity[-1])
    ax.set_xlabel(r'Velocity [km s$^{-1}$]', fontsize=10)
    ax.set_ylabel('Brightness temperature [K]', fontsize=10)
    ax.tick_params(labelsize=6)
    ax.set_title('CO(2-1) Spectrum', fontsize=10, fontweight='bold')

    ax2 = ax.twinx()
    ax2.plot(velocity, spectrum_mJy, color='k', alpha=0)
    ax2.set_ylabel('Flux Density [mJy]', fontsize=10, color='gray')
    ax2.tick_params(labelsize=6, colors='gray')
    ylim_K = ax.get_ylim()
    if (ylim_K[1]-ylim_K[0]) != 0:
        ratio = (np.nanmax(spectrum_mJy)-np.nanmin(spectrum_mJy))/(ylim_K[1]-ylim_K[0])
        ax2.set_ylim(ylim_K[0]*ratio, ylim_K[1]*ratio)


def create_summary_detected(galaxy, path, chans2do, spec_res=10,
                             savepath=None, overwrite=True):
    if savepath is None: savepath = path
    
    # 덮어쓰기 체크
    out_png = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + galaxy + '_summary.png'
    if not overwrite and os.path.exists(out_png):
        print(f'  Skipping {galaxy} (exists, overwrite=False)')
        return

    map_dir = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    
    # --- 핵심 수정: UL 맵을 기준으로 공통 extent 계산 ---
    common_extent = _get_data_extent(map_dir)

    map_configs = [
        ('*Ico_K_kms-1.fits', r'Moment 0 ($I_{\rm CO}$)', 'magma_r', r'$I_{\rm CO}$ [K km s$^{-1}$]', 0, None),
        ('*mmol_pc-2.fits', r'$\Sigma_{\rm mol}$', 'magma_r', r'$\Sigma_{\rm mol}$ [M$_\odot$ pc$^{-2}$]', 0, None),
        ('*peak_temp_k.fits', 'Peak Temperature', 'magma_r', r'$T_{\rm peak}$ [K]', 0, None),
        ('*mom1.fits', 'Velocity (Mom 1)', VELOCITY_CMAP, r'$v$ [km s$^{-1}$]', 'mom1', 'mom1'),
        ('*mom2.fits', r'$\sigma_v$ (Mom 2)', VELOCITY_CMAP, r'$\sigma_v$ [km s$^{-1}$]', 0, 'mom2'),
        ('*mom0_SN.fits', 'Signal-to-Noise', 'magma_r', 'S/N', 0, None),
        ('*Ico_K_kms-1_err.fits', r'$I_{\rm CO}$ Error', 'magma_r', r'$\sigma(I_{\rm CO})$ [K km s$^{-1}$]', 0, None),
        ('*Ico_K_kms-1_ul.fits', r'$I_{\rm CO}$ Upper Limit', 'magma_r', r'UL [K km s$^{-1}$]', None, None),
    ]

    # Velocity range logic
    clipping_table = pd.read_csv(chans2do)
    kgasid = int(galaxy.split('KGAS')[1])
    row = clipping_table[clipping_table['KGAS_ID'] == kgasid]
    vmin_mom1 = row['minchan_v'].iloc[0] if len(row) > 0 else None
    vmax_mom1 = row['maxchan_v'].iloc[0] if len(row) > 0 else None

    fig = plt.figure(figsize=(19, 16))
    fig.suptitle(f"{galaxy}  ({spec_res} km/s)", fontsize=16, fontweight='bold', y=0.98)

    ncols, nrows = 3, 3
    left_m, right_m, bot_m, top_m = 0.06, 0.02, 0.04, 0.06
    hspace, vspace = 0.09, 0.07
    panel_w = (1.0 - left_m - right_m - (ncols-1)*hspace) / ncols
    panel_h = (1.0 - bot_m - top_m - (nrows-1)*vspace) / nrows

    panel_idx = 0
    for r_idx in range(nrows):
        for c_idx in range(ncols):
            left = left_m + c_idx * (panel_w + hspace)
            bottom = 1.0 - top_m - (r_idx + 1) * panel_h - r_idx * vspace
            rect = [left, bottom, panel_w, panel_h]

            if panel_idx < len(map_configs):
                pattern, title, cmap, cbar_label, vmin_cfg, vmax_cfg = map_configs[panel_idx]
                files = glob(map_dir + pattern)
                if all(x not in pattern for x in ['_err', '_ul', '_SN']):
                    files = [f for f in files if all(x not in f for x in ['_err', '_ul', '_SN'])]

                if files:
                    hdu = fits.open(files[0])[0]
                    v_min = vmin_mom1 if vmin_cfg == 'mom1' else vmin_cfg
                    v_max = vmax_mom1 if vmax_cfg == 'mom1' else vmax_cfg
                    if vmax_cfg == 'mom2': 
                        v_max = min(100, np.nanmax(hdu.data[np.isfinite(hdu.data)]))

                    _add_map_panel(fig, rect, hdu, title, cmap=cmap, vmin=v_min, vmax=v_max,
                                   cbar_label=cbar_label, extent=common_extent)
                else:
                    ax = fig.add_axes(rect); ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
                    ax.set_title(title, fontsize=13, fontweight='bold')
            elif panel_idx == len(map_configs):
                ax = fig.add_axes(rect)
                _add_spectrum_panel(ax, galaxy, path, spec_res=spec_res)
            panel_idx += 1

    # Save

    outdir_gal = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    outdir_prod = savepath + 'by_product/summary/' + str(spec_res) + 'kms/'
    os.makedirs(outdir_gal, exist_ok=True)
    os.makedirs(outdir_prod, exist_ok=True)
    
    primary_png = outdir_gal + galaxy + '_summary.png'
    primary_pdf = outdir_gal + galaxy + '_summary.pdf'
    
    secondary_png = outdir_prod + galaxy + '_summary.png'
    secondary_pdf = outdir_prod + galaxy + '_summary.pdf'

    fig.savefig(primary_png, dpi=150, bbox_inches='tight')
    fig.savefig(primary_pdf, bbox_inches='tight')

    shutil.copy(primary_png, secondary_png)
    shutil.copy(primary_pdf, secondary_pdf)
    
    plt.close(fig)
    print(f'  Summary panel saved for {galaxy} ({spec_res} km/s)')


def create_summary_nondetected(galaxy, path, spec_res=10, savepath=None, overwrite=True):
    if savepath is None: savepath = path
    out_png = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + galaxy + '_summary.png'
    if not overwrite and os.path.exists(out_png):
        print(f'  Skipping {galaxy} (exists, overwrite=False)')
        return

    map_dir = path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    common_extent = _get_data_extent(map_dir)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"{galaxy}  ({spec_res} km/s) -- Non-detection", fontsize=14, fontweight='bold', y=0.97)

    rect_left = [0.06, 0.1, 0.4, 0.8]
    rect_right = [0.55, 0.1, 0.4, 0.8]

    ul_files = glob(map_dir + '*Ico_K_kms-1_ul.fits')
    if ul_files:
        _add_map_panel(fig, rect_left, fits.open(ul_files[0])[0], r'$I_{\rm CO}$ Upper Limit (3$\sigma$)',
                       cmap='magma_r', cbar_label=r'UL [K km s$^{-1}$]', extent=common_extent)
    
    ax_spec = fig.add_axes(rect_right)
    _add_spectrum_panel(ax_spec, galaxy, path, spec_res=spec_res)

    outdir_gal = savepath + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/'
    outdir_prod = savepath + 'by_product/summary/' + str(spec_res) + 'kms/'
    os.makedirs(outdir_gal, exist_ok=True)
    os.makedirs(outdir_prod, exist_ok=True)

    primary_png = outdir_gal + galaxy + '_summary.png'
    primary_pdf = outdir_gal + galaxy + '_summary.pdf'
    
    secondary_png = outdir_prod + galaxy + '_summary.png'
    secondary_pdf = outdir_prod + galaxy + '_summary.pdf'

    fig.savefig(primary_png, dpi=150, bbox_inches='tight')
    fig.savefig(primary_pdf, bbox_inches='tight')

    shutil.copy(primary_png, secondary_png)
    shutil.copy(primary_pdf, secondary_pdf)
    
    plt.close(fig)
    print(f'  Summary panel saved for {galaxy} ({spec_res} km/s, non-detection)')



    

def perform_summary_imaging(path, detections, non_detections, chans2do,
                             spec_res=10, savepath=None, overwrite=True):
    print(f'Creating summary panels ({spec_res} km/s)...')
    print(f'  Detected: {len(detections)}, Non-detected: {len(non_detections)}')
    for galaxy in detections:
        create_summary_detected(galaxy, path, chans2do, spec_res, savepath, overwrite)
    for galaxy in non_detections:
        create_summary_nondetected(galaxy, path, spec_res, savepath, overwrite)
    print('Summary panel creation complete.')

if __name__ == '__main__':
    spec_res = 10
    ifu_matched = False
    version = 3.0
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