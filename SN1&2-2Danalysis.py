# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:40:24 2026

@author: ekim
"""


from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd


# ==========================================================
# SETTINGS
# ==========================================================

folder = r"C:/Users/ekim/Documents/research/SN1&SN2/SN2"

fits_files = sorted(
    glob.glob(
        os.path.join(
            folder,
            "*.fits"
        )
    )
)

print(
    f"Found {len(fits_files)} FITS files"
)


# ==========================================================
# ZEMAX CSV FILES
# ==========================================================

# Diffraction-limited Zemax data

diffraction_csv = (
    r"C:/Users/ekim/Downloads/diffraction_limited_ensquared_energy.csv"
)


# Field Zemax data

field_csv = (
    r"C:/Users/ekim/Downloads/field_0_330mm_ensquared_energy.csv"
)


# ==========================================================
# ROI COORDINATES
# ==========================================================

# The original ROI was 8x10 pixels.
# An 11x11 EE aperture requires a ROI at least 11x11 pixels.

x_min = 5525
x_max = 5555

y_min = 4250
y_max = 4280


# ==========================================================
# DETECTOR SCALE
# ==========================================================

scale_um = 3.76


# ==========================================================
# ENSQUARED ENERGY APERTURES
# ==========================================================

# Only odd-sized squares are used.
# This keeps the same center pixel for every aperture.

square_sizes = [
    1,
    3,
    5,
    7,
    9,
    11
]


# ==========================================================
# ARRAYS TO STORE RESULTS
# ==========================================================

centroid_x = []
centroid_y = []

total_intensity = []

processed_files = []
all_rois = []

# Store background level for each image

background_levels = []


# Store EE for each aperture size

ee_results = {}

for size in square_sizes:

    ee_results[size] = []


# Store the center pixel selected for each image

center_pixels_x = []
center_pixels_y = []


# ==========================================================
# PROCESS EACH FITS IMAGE
# ==========================================================

for file in fits_files:

    # ------------------------------------------------------
    # LOAD FITS IMAGE
    # ------------------------------------------------------

    with fits.open(file) as hdul:

        image = hdul[0].data.astype(float)


    # ------------------------------------------------------
    # CROP ROI
    # ------------------------------------------------------

    roi = image[
        y_min:y_max,
        x_min:x_max
    ]


    # ======================================================
    # BACKGROUND SUBTRACTION
    # ======================================================

    # Estimate the background from the OUTER EDGE of the ROI.
    #
    # The background region is the 1-pixel-wide border:
    #
    #   X X X X X X X X X X X
    #   X                   X
    #   X                   X
    #   X                   X
    #   X                   X
    #   X                   X
    #   X X X X X X X X X X X
    #
    # The four edges are averaged together to obtain
    # one background signal for the entire ROI.

    top_edge = roi[0, :]

    bottom_edge = roi[-1, :]

    left_edge = roi[1:-1, 0]

    right_edge = roi[1:-1, -1]


    # Combine all outer-edge pixels

    background_pixels = np.concatenate([
        top_edge,
        bottom_edge,
        left_edge,
        right_edge
    ])


    # Average the outer-edge pixels

    background = np.mean(
        background_pixels
    )


    # Store background level

    background_levels.append(
        background
    )


    # ------------------------------------------------------
    # SUBTRACT BACKGROUND FROM ROI
    # ------------------------------------------------------

    roi_subtracted = (
        roi - background
    )



    # Store background-subtracted ROI

    all_rois.append(
        roi_subtracted
    )


    # ------------------------------------------------------
    # 2D WEIGHTED CENTROID
    # ------------------------------------------------------

    y_pixels, x_pixels = np.indices(
        roi_subtracted.shape
    )


    total = np.sum(
        roi_subtracted
    )


    # Calculate weighted centroid

    weighted_x = (
        np.sum(
            x_pixels * roi_subtracted
        )
        / total
    )

    weighted_y = (
        np.sum(
            y_pixels * roi_subtracted
        )
        / total
    )


    # Store centroid

    centroid_x.append(
        weighted_x
    )

    centroid_y.append(
        weighted_y
    )


    # Store total background-subtracted intensity

    total_intensity.append(
        total
    )


    processed_files.append(
        os.path.basename(file)
    )


    # ======================================================
    # FIND FIXED CENTER PIXEL
    # ======================================================

    # Round the weighted centroid to the nearest pixel.
    #
    # This pixel will be the center of EVERY aperture
    # for this image.

    center_x = int(
        np.round(weighted_x)
    )

    center_y = int(
        np.round(weighted_y)
    )


    # Store center pixel

    center_pixels_x.append(
        center_x
    )

    center_pixels_y.append(
        center_y
    )


    # ======================================================
    # ENSQUARED ENERGY
    # ======================================================

    for size in square_sizes:

        # --------------------------------------------------
        # Calculate how many pixels extend from the
        # center pixel in each direction.
        #
        # Example:
        #
        # 1x1 -> 0 pixels on each side
        # 3x3 -> 1 pixel on each side
        # 5x5 -> 2 pixels on each side
        # 7x7 -> 3 pixels on each side
        # --------------------------------------------------

        half_size = size // 2


        # --------------------------------------------------
        # Determine aperture boundaries
        # --------------------------------------------------

        x_start = (
            center_x - half_size
        )

        x_end = (
            center_x
            + half_size
            + 1
        )

        y_start = (
            center_y - half_size
        )

        y_end = (
            center_y
            + half_size
            + 1
        )


        # --------------------------------------------------
        # Extract the square
        # --------------------------------------------------

        square = roi_subtracted[
            y_start:y_end,
            x_start:x_end
        ]


        # --------------------------------------------------
        # Check that the square has the correct size
        # --------------------------------------------------

        if square.shape != (
            size,
            size
        ):

            print(
                f"WARNING: {size}x{size} aperture "
                f"does not fit in ROI for {file}"
            )

            ee = np.nan

        else:

            # ----------------------------------------------
            # Energy inside the square
            # ----------------------------------------------

            square_energy = np.sum(
                square
            )


            # ----------------------------------------------
            # Ensquared Energy
            #
            # Fraction of total background-subtracted ROI
            # light contained inside the square.
            # ----------------------------------------------

            ee = (
                square_energy
                / total
            )


        # Store EE

        ee_results[size].append(
            ee
        )


# ==========================================================
# CONVERT RESULTS TO NUMPY ARRAYS
# ==========================================================

centroid_x = np.array(
    centroid_x
)

centroid_y = np.array(
    centroid_y
)

total_intensity = np.array(
    total_intensity
)

background_levels = np.array(
    background_levels
)

center_pixels_x = np.array(
    center_pixels_x
)

center_pixels_y = np.array(
    center_pixels_y
)


for size in square_sizes:

    ee_results[size] = np.array(
        ee_results[size]
    )


# ==========================================================
# AVERAGE CENTROID
# ==========================================================

mean_x = np.mean(
    centroid_x
)

mean_y = np.mean(
    centroid_y
)


# ==========================================================
# CENTROID PRECISION
# ==========================================================

std_x_pixels = np.std(
    centroid_x
)

std_y_pixels = np.std(
    centroid_y
)

std_x_um = (
    std_x_pixels
    * scale_um
)

std_y_um = (
    std_y_pixels
    * scale_um
)


# ==========================================================
# PRINT CENTROID RESULTS
# ==========================================================

print()
print("==============================================")
print("2D WEIGHTED CENTROID")
print("==============================================")

print(
    f"Mean X centroid: "
    f"{mean_x:.6f} pixels"
)

print(
    f"Mean Y centroid: "
    f"{mean_y:.6f} pixels"
)

print()

print("==============================================")
print("CENTROID PRECISION")
print("==============================================")

print(
    f"X standard deviation: "
    f"{std_x_pixels:.6f} pixels"
)

print(
    f"Y standard deviation: "
    f"{std_y_pixels:.6f} pixels"
)

print(
    f"X precision: "
    f"{std_x_um:.6f} µm"
)

print(
    f"Y precision: "
    f"{std_y_um:.6f} µm"
)


# ==========================================================
# BACKGROUND LEVEL
# ==========================================================

mean_background = np.mean(
    background_levels
)

std_background = np.std(
    background_levels
)


print()
print("==============================================")
print("BACKGROUND SIGNAL")
print("==============================================")

print(
    f"Average background: "
    f"{mean_background:.2f}"
)

print(
    f"Background standard deviation: "
    f"{std_background:.2f}"
)


# ==========================================================
# TOTAL INTENSITY
# ==========================================================

mean_total = np.mean(
    total_intensity
)

std_total = np.std(
    total_intensity
)

total_variation_percent = (
    std_total
    / mean_total
) * 100


print()
print("==============================================")
print("TOTAL BACKGROUND-SUBTRACTED INTENSITY")
print("==============================================")

print(
    f"Mean total intensity: "
    f"{mean_total:.2f}"
)

print(
    f"Total intensity standard deviation: "
    f"{std_total:.2f}"
)

print(
    f"Total intensity variation: "
    f"{total_variation_percent:.3f}%"
)


# ==========================================================
# READ ZEMAX CSV FILES
# ==========================================================

# Load diffraction-limited data

diffraction_data = pd.read_csv(
    diffraction_csv
)


# Load field data

field_data = pd.read_csv(
    field_csv
)


# Extract square width

diffraction_width_um = (
    diffraction_data[
        "Square Width (um)"
    ].to_numpy()
)


field_width_um = (
    field_data[
        "Square Width (um)"
    ].to_numpy()
)


# Extract ensquared energy

diffraction_ee = (
    diffraction_data[
        "Ensquared Energy (%)"
    ].to_numpy()
)


field_ee = (
    field_data[
        "Ensquared Energy (%)"
    ].to_numpy()
)


# ==========================================================
# PRINT ZEMAX DATA INFORMATION
# ==========================================================

print()
print("==============================================")
print("ZEMAX REFERENCE DATA")
print("==============================================")

print(
    f"Diffraction-limited data points: "
    f"{len(diffraction_width_um)}"
)

print(
    f"Field data points: "
    f"{len(field_width_um)}"
)

print()

print(
    f"Diffraction EE at 10 µm: "
    f"{np.interp(10, diffraction_width_um, diffraction_ee):.3f}%"
)

print(
    f"Field EE at 10 µm: "
    f"{np.interp(10, field_width_um, field_ee):.3f}%"
)


# ==========================================================
# ENSQUARED ENERGY RESULTS
# ==========================================================

print()
print("==============================================")
print("ENSQUARED ENERGY")
print("==============================================")


average_ee = []

std_ee = []


for size in square_sizes:

    # Remove any NaN values

    valid_ee = ee_results[size][
        ~np.isnan(
            ee_results[size]
        )
    ]


    # Average EE

    mean_ee = np.mean(
        valid_ee
    )


    # Standard deviation

    ee_std = np.std(
        valid_ee
    )


    # Store results

    average_ee.append(
        mean_ee
    )

    std_ee.append(
        ee_std
    )


    # Physical width of aperture

    width_um = (
        size
        * scale_um
    )


    print()

    print(
        f"{size}x{size} pixels"
    )

    print(
        f"  Width: "
        f"{width_um:.2f} µm"
    )

    print(
        f"  Average EE: "
        f"{mean_ee * 100:.3f}%"
    )

    print(
        f"  EE standard deviation: "
        f"{ee_std * 100:.3f}%"
    )


# Convert to arrays

average_ee = np.array(
    average_ee
)

std_ee = np.array(
    std_ee
)


# ==========================================================
# RESULTS TABLE
# ==========================================================

results = pd.DataFrame({

    "File":
        processed_files,

    "Background":
        background_levels,

    "Centroid X (pixels)":
        centroid_x,

    "Centroid Y (pixels)":
        centroid_y,

    "Centroid X (um)":
        centroid_x * scale_um,

    "Centroid Y (um)":
        centroid_y * scale_um,

    "Center Pixel X":
        center_pixels_x,

    "Center Pixel Y":
        center_pixels_y,

    "Total Intensity":
        total_intensity
})


# Add EE results to table

for size in square_sizes:

    results[
        f"EE_{size}x{size}"
    ] = (
        ee_results[size]
        * 100
    )


# ==========================================================
# GRAPH 1
# AVERAGE 2D IMAGE OF THE DOT
# ==========================================================

average_roi = np.mean(
    all_rois,
    axis=0
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    average_roi,
    cmap="gray",
    origin="lower"
)


# Plot average weighted centroid

plt.scatter(
    mean_x,
    mean_y,
    marker="x",
    s=100,
    linewidths=2,
    label="Weighted centroid"
)


plt.xlabel(
    "X pixel"
)

plt.ylabel(
    "Y pixel"
)

plt.title(
    "Average 2D Dot\n"
    "Background Subtracted"
)


plt.colorbar(
    label="Background-Subtracted Intensity"
)


plt.legend()

plt.tight_layout()

plt.show()


# ==========================================================
# GRAPH 2
# CENTROID SCATTER PLOT
# ==========================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(
    centroid_x,
    centroid_y,
    s=60
)


# Plot average centroid

plt.scatter(
    mean_x,
    mean_y,
    marker="x",
    s=150,
    linewidths=2,
    label="Mean centroid"
)


plt.xlabel(
    "X centroid (pixels)"
)

plt.ylabel(
    "Y centroid (pixels)"
)

plt.title(
    "2D Centroid Repeatability"
)

plt.legend()

plt.axis(
    "equal"
)

plt.tight_layout()

plt.show()


# ==========================================================
# GRAPH 3
# CENTROID POSITION FOR EACH IMAGE
# ==========================================================

image_numbers = np.arange(
    1,
    len(centroid_x) + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    image_numbers,
    centroid_x,
    marker="o",
    label="X centroid"
)


plt.plot(
    image_numbers,
    centroid_y,
    marker="o",
    label="Y centroid"
)


plt.xlabel(
    "Image Number"
)

plt.ylabel(
    "Centroid Position (pixels)"
)

plt.title(
    "Centroid Position Across Images"
)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================================
# GRAPH 4
# TOTAL INTENSITY
# ==========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    image_numbers,
    total_intensity,
    marker="o"
)


plt.xlabel(
    "Image Number"
)

plt.ylabel(
    "Background-Subtracted Total Intensity"
)

plt.title(
    "Total Intensity Across Images"
)

plt.tight_layout()

plt.show()

# ==========================================================
# GRAPH 5
# EXPERIMENTAL ENSQUARED ENERGY VS. SQUARE WIDTH
# WITH ZEMAX REFERENCE CURVES
# ==========================================================

# Convert aperture sizes from pixels to micrometers

square_width_um = (
    np.array(square_sizes)
    * scale_um
)


plt.figure(figsize=(9, 7))


# ==========================================================
# EXPERIMENTAL DATA
# ==========================================================

# Plot average experimental EE with standard deviation
# error bars.

plt.errorbar(
    square_width_um,
    average_ee * 100,
    yerr=std_ee * 100,
    marker="o",
    capsize=5,
    linewidth=2,
    label="Experimental"
)


# ==========================================================
# ZEMAX DIFFRACTION-LIMITED CURVE
# ==========================================================

plt.plot(
    diffraction_width_um,
    diffraction_ee,
    linewidth=2,
    label="Zemax Diffraction Limited"
)


# ==========================================================
# ZEMAX FIELD CURVE
# ==========================================================

plt.plot(
    field_width_um,
    field_ee,
    linewidth=2,
    label="Zemax Field: 0.00, 330.00 mm"
)


# ==========================================================
# AXIS LABELS
# ==========================================================

plt.xlabel(
    "Square Width (µm)"
)

plt.ylabel(
    "Percentage of Light Ensquared (%)"
)


plt.title(
    "Ensquared Energy vs. Square Width"
)


# ==========================================================
# X-AXIS TICKS
# ==========================================================

# Use the experimental physical widths as the x-axis ticks

plt.xticks(
    square_width_um,
    [
        f"{size}x{size}\n({width:.2f} µm)"
        for size, width in zip(
            square_sizes,
            square_width_um
        )
    ]
)


# ==========================================================
# AXIS LIMITS
# ==========================================================

# Only show the region containing the experimental data.
# This removes the large blank space to the right.

x_max_plot = (
    np.max(square_width_um) + 4
)

plt.xlim(
    0,
    x_max_plot
)


# Keep the EE range from 0 to 100%

plt.ylim(
    0,
    100
)


# ==========================================================
# GRID AND LEGEND
# ==========================================================

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()
# ==========================================================
# GRAPH 6
# VISUALIZE THE DIFFERENT APERTURES
# ==========================================================

# Use the first image

roi = all_rois[0]


# Get centroid from first image

x_center = center_pixels_x[0]

y_center = center_pixels_y[0]

x_centroid_first = centroid_x[0]

y_centroid_first = centroid_y[0]


plt.figure(
    figsize=(8, 8)
)


plt.imshow(
    roi,
    cmap="gray",
    origin="lower"
)


# ----------------------------------------------------------
# Plot weighted centroid
# ----------------------------------------------------------

plt.scatter(
    x_centroid_first,
    y_centroid_first,
    marker="+",
    s=150,
    linewidths=2,
    label="Weighted centroid"
)


# ----------------------------------------------------------
# Draw each aperture
# ----------------------------------------------------------

for size in square_sizes:

    half_size = (
        size // 2
    )


    # Determine pixel boundaries

    x_start = (
        x_center
        - half_size
    )

    x_end = (
        x_center
        + half_size
        + 1
    )

    y_start = (
        y_center
        - half_size
    )

    y_end = (
        y_center
        + half_size
        + 1
    )


    # ------------------------------------------------------
    # IMPORTANT:
    # The -0.5 values place the lines on the EDGES
    # of the pixels rather than at their centers.
    # ------------------------------------------------------

    rectangle_x = [

        x_start - 0.5,

        x_end - 0.5,

        x_end - 0.5,

        x_start - 0.5,

        x_start - 0.5

    ]


    rectangle_y = [

        y_start - 0.5,

        y_start - 0.5,

        y_end - 0.5,

        y_end - 0.5,

        y_start - 0.5

    ]


    plt.plot(
        rectangle_x,
        rectangle_y,
        linewidth=2,
        label=f"{size}x{size}"
    )


plt.xlabel(
    "X pixel"
)

plt.ylabel(
    "Y pixel"
)

plt.title(
    "Ensquared Energy Apertures\n"
    "Background Subtracted"
)


plt.colorbar(
    label="Background-Subtracted Intensity"
)


plt.legend()

plt.tight_layout()

plt.show()
