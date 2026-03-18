import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

try:
    from sauron_colormap import register_sauron_colormap
    register_sauron_colormap()
    VELOCITY_CMAP = 'sauron'
except:
    VELOCITY_CMAP = 'coolwarm'


def _add_beam(ax, header):
    """왼쪽 하단에 빔 그리기"""
    try:
        bmaj, bmin, bpa = header['BMAJ'], header['BMIN'], header.get('BPA', 0)
        pixscale = abs(header['CDELT2'])
        bmaj_pix, bmin_pix = bmaj / pixscale, bmin / pixscale
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        corner_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        corner_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
        
        beam = Ellipse((corner_x, corner_y), bmin_pix, bmaj_pix, angle=bpa,
                        edgecolor='black', facecolor='none', linewidth=1.0,
                        transform=ax.get_transform('pixel'))
        ax.add_patch(beam)
    except:
        pass


def plot_single_moment_grid(galaxies, path, chans2do_df, spec_res=10, moment='mom0', savepath='./', data_type='matched'):
    print(f"Creating Giant Grid for {moment} ({len(galaxies)} galaxies) - {data_type}...")
    
    if moment == 'mom0':
        pattern, cmap = '*Ico_K_kms-1.fits', 'magma_r'
    elif moment == 'mom1':
        pattern, cmap = '*mom1.fits', VELOCITY_CMAP
    elif moment == 'mom2':
        pattern, cmap = '*mom2.fits', VELOCITY_CMAP
    else:
        raise ValueError("moment must be 'mom0', 'mom1', or 'mom2'")

    n_gals = len(galaxies)
    ncols = int(np.ceil(np.sqrt(n_gals)))
    nrows = int(np.ceil(n_gals / ncols))
    
    fig = plt.figure(figsize=(max(ncols*3, 20), max(nrows*3, 20)))
    fig.suptitle(f"All Galaxies - {moment.upper()} ({spec_res} km/s)", fontsize=40, fontweight='bold', y=0.99)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for idx, galaxy in enumerate(galaxies):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        
        # CSV에서 mom1을 위한 속도 범위(vmin, vmax) 추출
        kgasid = int(galaxy.replace('KGAS', ''))
        row = chans2do_df[chans2do_df['KGAS_ID'] == kgasid]
        if not row.empty:
            v_0 = row['minchan_v'].iloc[0]
            v_1 = row['maxchan_v'].iloc[0]
            vmin_mom1 = v_0 if pd.notna(v_0) else None
            vmax_mom1 = v_1 if pd.notna(v_1) else None
        else:
            vmin_mom1, vmax_mom1 = None, None

        map_dir = os.path.join(path, 'by_galaxy', galaxy, f'{spec_res}kms')
        files = glob.glob(os.path.join(map_dir, pattern))
        if '_err' not in pattern and '_ul' not in pattern:
            files = [f for f in files if '_err' not in f and '_ul' not in f]

        if files:
            try:
                hdu = fits.open(files[0])[0]
                data, header = hdu.data, hdu.header
                wcs_full = WCS(header, naxis=2)
                
                ax.remove()
                ax = fig.add_subplot(nrows, ncols, idx + 1, projection=wcs_full)
                
                finite_idx = np.where(np.isfinite(data))
                if len(finite_idx[0]) == 0:
                    ax.text(0.5, 0.5, 'NaN\nData', transform=ax.transAxes, ha='center', va='center')
                else:
                    vmin, vmax = np.nanmin(data[finite_idx]), np.nanmax(data[finite_idx])
                    
                    # 💡 Moment 1과 2에 대한 컬러 스케일 고정 로직
                    if moment == 'mom1':
                        vmin = vmin_mom1 if vmin_mom1 is not None else vmin
                        vmax = vmax_mom1 if vmax_mom1 is not None else vmax
                    elif moment == 'mom2': 
                        vmax = min(100, vmax)
                    
                    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                                   interpolation='nearest', aspect='equal')
                    
                    pad = 2
                    y_min, y_max = np.min(finite_idx[0]), np.max(finite_idx[0])
                    x_min, x_max = np.min(finite_idx[1]), np.max(finite_idx[1])
                    ax.set_xlim(x_min - pad, x_max + pad)
                    ax.set_ylim(y_min - pad, y_max + pad)
                    
                    _add_beam(ax, header)
            except Exception as e:
                print(f"  Error on {galaxy}: {e}")
                ax.text(0.5, 0.5, 'Error', transform=ax.transAxes, ha='center', va='center', color='red')
        else:
            ax.text(0.5, 0.5, 'No File', transform=ax.transAxes, ha='center', va='center', color='gray')

        ax.set_title(galaxy, fontsize=10, fontweight='bold')
        ax.axis('off')

    out_pdf = os.path.join(savepath, f"All_Galaxies_{moment.upper()}_{spec_res}kms_{data_type}.pdf")
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved giant grid to: {out_pdf}")


def plot_side_by_side_grid(galaxies, path, chans2do_df, spec_res=10, savepath='./', data_type='matched'):
    print(f"Creating Giant Side-by-Side Grid (Mom0 & Mom1) for {len(galaxies)} galaxies - {data_type}...")
    
    n_gals = len(galaxies)
    ncols_gals = int(np.ceil(np.sqrt(n_gals)))
    nrows = int(np.ceil(n_gals / ncols_gals))
    
    fig = plt.figure(figsize=(max(ncols_gals*5, 30), max(nrows*3, 20)))
    fig.suptitle(f"All Galaxies - Mom0(Left) & Mom1(Right) ({spec_res} km/s)", fontsize=40, fontweight='bold', y=0.99)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    for idx, galaxy in enumerate(galaxies):
        
        # CSV에서 mom1 속도 범위 추출
        kgasid = int(galaxy.replace('KGAS', ''))
        row = chans2do_df[chans2do_df['KGAS_ID'] == kgasid]
        if not row.empty:
            v_0 = row['minchan_v'].iloc[0]
            v_1 = row['maxchan_v'].iloc[0]
            vmin_mom1 = v_0 if pd.notna(v_0) else None
            vmax_mom1 = v_1 if pd.notna(v_1) else None
        else:
            vmin_mom1, vmax_mom1 = None, None

        map_dir = os.path.join(path, 'by_galaxy', galaxy, f'{spec_res}kms')
        
        extent = None
        mom0_files = glob.glob(os.path.join(map_dir, '*Ico_K_kms-1.fits'))
        mom0_files = [f for f in mom0_files if '_err' not in f and '_ul' not in f]
        if mom0_files:
            try:
                data_mom0 = fits.getdata(mom0_files[0])
                finite_idx = np.where(np.isfinite(data_mom0))
                if len(finite_idx[0]) > 0:
                    pad = 2
                    y_min, y_max = np.min(finite_idx[0]), np.max(finite_idx[0])
                    x_min, x_max = np.min(finite_idx[1]), np.max(finite_idx[1])
                    extent = [x_min - pad, x_max + pad, y_min - pad, y_max + pad]
            except:
                pass
        
        row_idx = idx // ncols_gals
        col_idx = idx % ncols_gals
        pos_left = row_idx * (ncols_gals * 2) + (col_idx * 2) + 1
        pos_right = pos_left + 1

        for i, (moment, pattern, cmap) in enumerate([
            ('mom0', '*Ico_K_kms-1.fits', 'magma_r'),
            ('mom1', '*mom1.fits', VELOCITY_CMAP)
        ]):
            pos = pos_left if i == 0 else pos_right
            ax = fig.add_subplot(nrows, ncols_gals * 2, pos)
            
            files = glob.glob(os.path.join(map_dir, pattern))
            files = [f for f in files if '_err' not in f and '_ul' not in f]

            if files:
                try:
                    hdu = fits.open(files[0])[0]
                    data, header = hdu.data, hdu.header
                    
                    ax.remove()
                    ax = fig.add_subplot(nrows, ncols_gals * 2, pos, projection=WCS(header, naxis=2))
                    
                    finite = data[np.isfinite(data)]
                    if len(finite) > 0:
                        vmin, vmax = np.nanmin(finite), np.nanmax(finite)
                        
                        # 💡 Mom1의 경우 CSV의 범위를 우선 적용
                        if moment == 'mom1':
                            vmin = vmin_mom1 if vmin_mom1 is not None else vmin
                            vmax = vmax_mom1 if vmax_mom1 is not None else vmax
                            
                        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                                       interpolation='nearest', aspect='equal')
                        
                        if extent:
                            ax.set_xlim(extent[0], extent[1])
                            ax.set_ylim(extent[2], extent[3])
                            
                        _add_beam(ax, header)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error', transform=ax.transAxes, ha='center', va='center', color='red')
            else:
                ax.text(0.5, 0.5, 'No File', transform=ax.transAxes, ha='center', va='center', color='gray')

            title_text = f"{galaxy} ({moment})"
            ax.set_title(title_text, fontsize=8, fontweight='bold')
            ax.axis('off')

    out_pdf = os.path.join(savepath, f"All_Galaxies_Side_by_Side_{spec_res}kms_{data_type}.pdf")
    os.makedirs(savepath, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved side-by-side giant grid to: {out_pdf}")

if __name__ == '__main__':
    # ── 설정 ──
    spec_res = 30
    version = 1.2
    ifu_matched = False
    
    data_type = "matched" if ifu_matched else "original"
    path = f"/arc/projects/KILOGAS/products/v{version}/{data_type}/"
    chans2do = "KGAS_chans2do_v_optical_Sept25.csv"

    # CSV 데이터 로드 (모멘트 1 속도 클리핑용)
    chans2do_df = pd.read_csv(chans2do)

    target_id = chans2do_df.iloc[:, 0].values
    col = 7 if spec_res == 30 else 6
    detected = chans2do_df.iloc[:, col].values
    
    detections = ['KGAS' + str(t) for t, d in zip(target_id, detected) if d]

    #targets = ['KGAS10', 'KGAS12', 'KGAS13', 'KGAS15', 'KGAS16', 'KGAS17', 'KGAS19', 'KGAS20', 'KGAS21', 'KGAS22', 'KGAS23', 'KGAS24', 'KGAS25', 'KGAS26', 'KGAS27', 'KGAS28']
    #detections = [g for g in targets if g in detections]
    
    savepath = f"/arc/projects/KILOGAS/products/v{version}/{data_type}/summary_grids/"
    
    # ── 1. 단일 모멘트 선택형 전체 그림 ──
    plot_single_moment_grid(detections, path, chans2do_df, spec_res=spec_res, moment='mom0', savepath=savepath, data_type=data_type)
    #plot_single_moment_grid(detections, path, chans2do_df, spec_res=spec_res, moment='mom1', savepath=savepath, data_type=data_type)
    
    # ── 2. Mom0 & Mom1 나란히(Side-by-Side) 그리기 ──
    plot_side_by_side_grid(detections, path, chans2do_df, spec_res=spec_res, savepath=savepath, data_type=data_type)