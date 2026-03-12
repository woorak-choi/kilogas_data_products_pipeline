"""
check_beam_sizes.py

Compares beam sizes (BMAJ, BMIN, BPA) between 10 km/s and 30 km/s 
upper limit products for all galaxies. Flags galaxies where the beam 
is identical between the two spectral resolutions, which may indicate 
that the wrong cube was used or files were copied incorrectly.

Usage:
    python check_beam_sizes.py

Output:
    - Console summary of all galaxies with matching/mismatching beams
    - CSV file with full comparison table
    
Author: Woorak Choi
Date: March 2026
"""

import warnings
warnings.filterwarnings("ignore")

from astropy.io import fits
from glob import glob
import numpy as np
import os
import csv


def check_beam_sizes(
    path,
    spec_res_list=[10, 30],
    pattern="*Ico_K_kms-1_ul.fits",
    save_csv=True,
):
    """
    Compare beam sizes between different spectral resolutions.
    
    Parameters
    ----------
    path : str
        Base path to products (e.g., /arc/projects/KILOGAS/products/v1.1/matched/)
    spec_res_list : list
        Two spectral resolutions to compare [10, 30].
    pattern : str
        Glob pattern for the FITS files to check.
    save_csv : bool
        If True, save results to a CSV file.
    """
    
    res_a, res_b = spec_res_list
    
    # Find all galaxies that have products in BOTH resolutions
    galaxies_a = set()
    galaxies_b = set()
    
    for gal_dir in sorted(glob(path + "by_galaxy/KGAS*")):
        galaxy = os.path.basename(gal_dir)
        
        files_a = glob(gal_dir + "/" + str(res_a) + "kms/" + pattern)
        files_b = glob(gal_dir + "/" + str(res_b) + "kms/" + pattern)
        
        if files_a:
            galaxies_a.add(galaxy)
        if files_b:
            galaxies_b.add(galaxy)
    
    common = sorted(galaxies_a & galaxies_b)
    only_a = sorted(galaxies_a - galaxies_b)
    only_b = sorted(galaxies_b - galaxies_a)
    
    print("=" * 70)
    print(f"Beam Size Comparison: {res_a} km/s vs {res_b} km/s")
    print(f"Pattern: {pattern}")
    print(f"Path: {path}")
    print("=" * 70)
    print(f"Galaxies with both resolutions: {len(common)}")
    print(f"Galaxies with {res_a} km/s only: {len(only_a)}")
    print(f"Galaxies with {res_b} km/s only: {len(only_b)}")
    print()
    
    # Compare beam sizes
    results = []
    identical_beams = []
    different_beams = []
    errors = []
    
    for galaxy in common:
        file_a = glob(path + "by_galaxy/" + galaxy + "/" + str(res_a) + "kms/" + pattern)
        file_b = glob(path + "by_galaxy/" + galaxy + "/" + str(res_b) + "kms/" + pattern)
        
        try:
            hdr_a = fits.getheader(file_a[0])
            hdr_b = fits.getheader(file_b[0])
            
            bmaj_a = hdr_a.get('BMAJ', np.nan)
            bmin_a = hdr_a.get('BMIN', np.nan)
            bpa_a = hdr_a.get('BPA', np.nan)
            
            bmaj_b = hdr_b.get('BMAJ', np.nan)
            bmin_b = hdr_b.get('BMIN', np.nan)
            bpa_b = hdr_b.get('BPA', np.nan)
            
            # Convert to arcsec for display
            bmaj_a_arcsec = bmaj_a * 3600 if np.isfinite(bmaj_a) else np.nan
            bmin_a_arcsec = bmin_a * 3600 if np.isfinite(bmin_a) else np.nan
            bmaj_b_arcsec = bmaj_b * 3600 if np.isfinite(bmaj_b) else np.nan
            bmin_b_arcsec = bmin_b * 3600 if np.isfinite(bmin_b) else np.nan
            
            # Check if beams are identical
            bmaj_match = (bmaj_a == bmaj_b) or (np.isnan(bmaj_a) and np.isnan(bmaj_b))
            bmin_match = (bmin_a == bmin_b) or (np.isnan(bmin_a) and np.isnan(bmin_b))
            bpa_match = (bpa_a == bpa_b) or (np.isnan(bpa_a) and np.isnan(bpa_b))
            
            all_match = bmaj_match and bmin_match and bpa_match
            
            # Also check with small tolerance (floating point)
            bmaj_close = np.isclose(bmaj_a, bmaj_b, rtol=1e-6) if (np.isfinite(bmaj_a) and np.isfinite(bmaj_b)) else bmaj_match
            bmin_close = np.isclose(bmin_a, bmin_b, rtol=1e-6) if (np.isfinite(bmin_a) and np.isfinite(bmin_b)) else bmin_match
            
            all_close = bmaj_close and bmin_close
            
            row = {
                'galaxy': galaxy,
                f'BMAJ_{res_a}kms_arcsec': round(bmaj_a_arcsec, 4),
                f'BMIN_{res_a}kms_arcsec': round(bmin_a_arcsec, 4),
                f'BPA_{res_a}kms_deg': round(bpa_a, 2) if np.isfinite(bpa_a) else 'N/A',
                f'BMAJ_{res_b}kms_arcsec': round(bmaj_b_arcsec, 4),
                f'BMIN_{res_b}kms_arcsec': round(bmin_b_arcsec, 4),
                f'BPA_{res_b}kms_deg': round(bpa_b, 2) if np.isfinite(bpa_b) else 'N/A',
                'BMAJ_ratio': round(bmaj_a / bmaj_b, 4) if (np.isfinite(bmaj_a) and np.isfinite(bmaj_b) and bmaj_b != 0) else 'N/A',
                'identical': all_close,
            }
            results.append(row)
            
            if all_close:
                identical_beams.append(galaxy)
            else:
                different_beams.append(galaxy)
                
        except Exception as e:
            errors.append((galaxy, str(e)))
    
    # ── Print results ──
    print("-" * 70)
    print(f"IDENTICAL beams ({res_a} vs {res_b} km/s): {len(identical_beams)} galaxies")
    print("-" * 70)
    if identical_beams:
        for galaxy in identical_beams:
            row = [r for r in results if r['galaxy'] == galaxy][0]
            print(f"  {galaxy:10s}  BMAJ={row[f'BMAJ_{res_a}kms_arcsec']:8.4f}\"  "
                  f"BMIN={row[f'BMIN_{res_a}kms_arcsec']:8.4f}\"")
    else:
        print("  None")
    
    print()
    print("-" * 70)
    print(f"DIFFERENT beams ({res_a} vs {res_b} km/s): {len(different_beams)} galaxies")
    print("-" * 70)
    if different_beams:
        for galaxy in different_beams:
            row = [r for r in results if r['galaxy'] == galaxy][0]
            print(f"  {galaxy:10s}  "
                  f"{res_a}kms: {row[f'BMAJ_{res_a}kms_arcsec']:8.4f}\" x {row[f'BMIN_{res_a}kms_arcsec']:8.4f}\"  |  "
                  f"{res_b}kms: {row[f'BMAJ_{res_b}kms_arcsec']:8.4f}\" x {row[f'BMIN_{res_b}kms_arcsec']:8.4f}\"  |  "
                  f"ratio: {row['BMAJ_ratio']}")
    else:
        print("  None")
    
    if errors:
        print()
        print("-" * 70)
        print(f"ERRORS: {len(errors)} galaxies")
        print("-" * 70)
        for galaxy, err in errors:
            print(f"  {galaxy}: {err}")
    
    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print(f"  Total compared:     {len(common)}")
    print(f"  Identical beams:    {len(identical_beams)}  *** FLAGGED ***")
    print(f"  Different beams:    {len(different_beams)}  (expected)")
    print(f"  Errors:             {len(errors)}")
    print("=" * 70)
    
    # ── Save CSV ──
    if save_csv and results:
        csv_path = path + f"beam_comparison_{res_a}vs{res_b}kms.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nFull table saved to: {csv_path}")
    
    return results, identical_beams, different_beams


if __name__ == '__main__':
    ifu_matched = False
    version = 3.0

    if ifu_matched:
        path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
        path = "/arc/home/rock211/test_products/" + str(version) + "/matched/"
    else:
        path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"
        path = "/arc/home/rock211/test_products/" + str(version) + "/original/"

    # Check upper limit products
    print("\n### Checking upper limit (UL) products ###\n")
    results_ul, identical_ul, different_ul = check_beam_sizes(
        path=path,
        spec_res_list=[10, 30],
        pattern="*Ico_K_kms-1_ul.fits",
    )

    # Optionally also check other products:
    # print("\n\n### Checking Ico products ###\n")
    # results_ico, identical_ico, different_ico = check_beam_sizes(
    #     path=path,
    #     spec_res_list=[10, 30],
    #     pattern="*Ico_K_kms-1.fits",
    # )
    #
    # print("\n\n### Checking error products ###\n")
    # results_err, identical_err, different_err = check_beam_sizes(
    #     path=path,
    #     spec_res_list=[10, 30],
    #     pattern="*Ico_K_kms-1_err.fits",
    # )
