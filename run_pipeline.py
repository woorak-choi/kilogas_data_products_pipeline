import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False 
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # ← 추가

import smooth_and_clip
import create_moments
import image_moments
import create_spectrum
import numpy as np
import warnings
import create_summary_panel


warnings.filterwarnings("ignore")
# from glob import glob
import os
import shutil


if __name__ == "__main__":
    # Set parameters deciding which products are made for which cubes
    ifu_match = True
    local = False
    clear_save_directory = False
    version = 3.0
    spec_res = 30
    pb_thresh = 40
    prune_by_npix = None

    print ('version=',version)
    print ('spectral resolution [km/s]', spec_res)
    print ('ifu_match',ifu_match)
    # targets = [d.split('/')[-1] for d in glob(main_directory + '*') if os.path.isdir(d) and os.listdir(d)]

    # Set paths for either running the script locally or on Canfar
    if local:
        main_directory = ""
        save_path = main_directory
        # detected = np.genfromtxt('')
        # target_id = np.genfromtxt('')
        chans2do = ""
        glob_cat = ""
    else:
        if ifu_match:
            main_directory = "/arc/projects/KILOGAS/cubes/v1.0/matched/"
            save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
            save_path = "/arc/home/rock211/test_products/" + str(version) + "/matched/"
        else:
            main_directory = "/arc/projects/KILOGAS/cubes/v1.0/nyquist/"
            save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"
            save_path = "/arc/home/rock211/test_products/" + str(version) + "/original/"
        

        chans2do = "KGAS_chans2do_v_optical_Sept25.csv"
        if spec_res == 10:
            detected = np.genfromtxt(
                chans2do, delimiter=",", skip_header=1, usecols=[6], dtype=bool
            )
        elif spec_res == 30:
            detected = np.genfromtxt(
                chans2do, delimiter=",", skip_header=1, usecols=[7], dtype=bool
            )
        target_id = np.genfromtxt(
            chans2do, delimiter=",", skip_header=1, usecols=[0], dtype=int
        )
        glob_cat = "KILOGAS_global_catalog_FWHM.fits"

    
    # Create lists of detections, non detections, and all targets depending on
    # how they are flagged in the global table.
    detections = [
        "KGAS" + str(target) for target, flag in zip(target_id, detected) if flag
    ]
    non_detections = [
        "KGAS" + str(target) for target, flag in zip(target_id, detected) if not flag
    ]
    targets = ["KGAS" + str(target) for target in target_id]

    # # ── Custom override ──
    # targets = ['KGAS361']
    
    # # targets 중에서 자동으로 detection/non-detection 분류
    # detections = [g for g in targets if g in detections]
    # non_detections = [g for g in targets if g in non_detections]


    
    # The above lists can be manually overwritten for debugging purposes or in
    # case products are made for specific galaxies
    # targets = ['KGAS107']
    # detections = ['KGAS107']
    # non_detections = []

    # The below will empty the overall directory in which the products are saved
    # before storing the new products there.
    if clear_save_directory:
        for galaxy in non_detections:
            directory = os.path.join(save_path, galaxy)
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.unlink(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    shutil.rmtree(dir_path)

    # The below calls the script to make the various products. Toggle on and off
    # as desired. Note that S+C will have to be done before creating moments, but
    # spectra can be created independently.

    print("\n" + "="*60)
    print("STEP 1/5: Smooth and Clip")
    print(f"  Targets: {len(detections)} detected galaxies")
    print("="*60)

    smooth_and_clip.perform_smooth_and_clip(
        read_path=main_directory,
        save_path=save_path,
        targets=detections,
        chans2do=chans2do,
        kms=spec_res,
        pb_thresh=pb_thresh,
        prune_by_npix=prune_by_npix,
        ifu_match=ifu_match,
    )


    print("\n" + "="*60)
    print("STEP 2/5: Moment Maps, Uncertainties, and Upper Limits")
    print(f"  Targets: {len(detections)} detected + {len(non_detections)} non-detected")
    print("="*60)

    create_moments.perform_moment_creation(
        path=save_path,
        data_path=main_directory,
        detections=detections,
        non_detections=non_detections,
        glob_cat=glob_cat,
        spec_res=spec_res,
        ifu_match=ifu_match,
        pb_thresh=pb_thresh,
    )


    print("\n" + "="*60)
    print("STEP 3/5: Moment Map Imaging")
    print(f"  Targets: {len(detections)} detected galaxies")
    print("="*60)

    image_moments.perform_moment_imaging(
        glob_path=save_path, targets=detections, chans2do=chans2do, spec_res=spec_res
    )

    print("\n" + "="*60)
    print("STEP 4/5: Spectrum Extraction")
    print(f"  Targets: {len(targets)} galaxies")
    print("="*60)
    
    create_spectrum.get_all_spectra(
        read_path=main_directory,
        save_path=save_path,
        targets=targets,
        target_id=target_id,
        detected=detected,
        chans2do=chans2do,
        glob_cat=glob_cat,
        spec_res=spec_res,
    )

    print("\n" + "="*60)
    print("STEP 5/5: Summary Panels")
    print(f"  Targets: {len(detections)} detected + {len(non_detections)} non-detected")
    print("="*60)
    
    create_summary_panel.perform_summary_imaging(
    path=save_path,
    detections=detections,
    non_detections=non_detections,
    chans2do=chans2do,
    spec_res=spec_res,
    )
