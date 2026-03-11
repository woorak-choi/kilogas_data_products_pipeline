네, 요청하신 대로 KILOGAS 파이프라인 매뉴얼의 영문 버전을 작성해 드립니다. README 파일이나 영문 위키 문서로 바로 활용하실 수 있도록 구성했습니다.

---

# 🌌 KILOGAS Data Products Pipeline User Manual

## 1. Overview

The KILOGAS Data Products Pipeline is an automated tool designed to process ALMA CO(2-1) data cubes. It generates a comprehensive set of data products, including moment maps, spectra, uncertainty/upper-limit maps, and final summary panels.

The entire pipeline is controlled and executed through a single main script: **`run_pipeline.py`**.

---

## 2. Configuration & Execution

Before running the pipeline, you must configure the primary parameters at the top of the `run_pipeline.py` script to match your current analysis goals.

**Key Parameters:**

* `version` (e.g., `2.0`): Defines the output directory version where products will be saved.
* `spec_res` (e.g., `10` or `30`): Selects the spectral resolution (in km/s) of the input data cubes.
* `ifu_match` (`True` / `False`): Determines whether to use IFU-matched cubes (`True`) or original/nyquist cubes (`False`).
* `clear_save_directory` (`True` / `False`): If `True`, completely deletes and resets the output directory before generating new products.

**How to Run:**
Execute the following command in your terminal:

```bash
python run_pipeline.py

```

---

## 3. The 5 Pipeline Steps

Once executed, `run_pipeline.py` runs through the following 5 sequential steps:

1. **STEP 1: Smooth and Clip (`smooth_and_clip.py`)**
* Uses the Dame et al. (2011) method to separate noise and create a signal mask.
* Outputs `_clipped_cube.fits` and `_mask_cube.fits`. *(Note: This is the most computationally intensive and time-consuming step).*


2. **STEP 2: Moment Map Creation (`create_moments.py`)**
* Generates Moment 0, 1, and 2 maps based on the galaxy's detection status.
* Creates maps in various units (K km/s, Msol/pc², Msol/pix) along with their corresponding error and 3σ upper limit maps.


3. **STEP 3: Moment Map Imaging (`image_moments.py`)**
* Renders the generated FITS files into high-quality PNG and PDF images.
* Automatically overlays contour lines, color bars, and synthesized beam sizes.


4. **STEP 4: Spectrum Extraction (`create_spectrum.py`)**
* Extracts the CO spectrum from the detected emission area (or within the R50 radius for non-detections).
* Saves the data as a CSV file and plots Brightness Temperature (K) & Flux Density (mJy) vs. Velocity (km/s).


5. **STEP 5: Summary Panels (`create_summary_panel.py`)**
* Compiles all individual visual products into a single summary figure per galaxy.
* **Detections:** 3x3 grid (8 maps + 1 spectrum plot).
* **Non-detections:** 1x2 grid (Upper limit map + spectrum plot).



---

## 4. Advanced Tips & Troubleshooting

### 💡 Processing Specific Galaxies (Custom Override)

If you need to re-run the pipeline for specific galaxies (e.g., for debugging or manual review) without processing the entire dataset, you can override the target list in `run_pipeline.py`:

```python
# 1. Manually override the target list
targets = ['KGAS10', 'KGAS105', 'KGAS108']

# 2. IMPORTANT: Update the detection lists based on the new targets
detections = [g for g in targets if g in detections]
non_detections = [g for g in targets if g in non_detections]

```

### ⏱️ Saving Time (Skipping Step 1)

If the masking and clipping process (Step 1) has already been successfully completed and you only need to regenerate images, moment maps, or spectra, you can significantly reduce execution time by skipping Step 1.

Simply comment out the `perform_smooth_and_clip` function call near the bottom of `run_pipeline.py`:

```python
    # print("STEP 1/5: Smooth and Clip")
    # smooth_and_clip.perform_smooth_and_clip(...) # <- Comment this out

```

The pipeline will bypass the heavy cube processing and directly use the existing `_clipped_cube.fits` to quickly generate the remaining products.
