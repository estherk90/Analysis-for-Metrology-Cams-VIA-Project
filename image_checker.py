# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:30:30 2026

@author: ekim
"""
#CHECKS REGION OF INTEREST

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# %%


file = r"C:/Users/ekim/Documents/research/SN1&SN2/SN2/focusing_5171mm_Comp_1ms_0001.fits"
with fits.open(file) as hdul:
    image = hdul[0].data.astype(float)



# PRINT IMAGE INFORMATION


print("Image shape:", image.shape)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())


# ==========================================================
# FULL IMAGE
# ==========================================================

# ==========================================================
# FULL IMAGE
# ==========================================================

plt.figure(figsize=(10,10))

plt.imshow(
    image,
    cmap="gray",
    origin="lower",
    vmin=np.percentile(image, 5),
    vmax=np.percentile(image, 99)
)
# %%


plt.colorbar(label="Pixel Value")

plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Full FITS Image")

plt.show()










# ==========================================================
# ZOOMED REGION OF INTEREST
# CHANGE THESE VALUES
# ==========================================================

x_min = 5534
x_max = 5542

y_min = 4260
y_max = 4270

roi = image[y_min:y_max, x_min:x_max]

print(roi.max())


# ==========================================================
# DISPLAY ROI
# ==========================================================

plt.figure(figsize=(8,8))

plt.imshow(
    roi,
    cmap="gray",
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    vmin=image.min(),
    vmax=image.max()
)

plt.colorbar(label="Pixel Value")

plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Zoomed Region of Interest")

plt.show()


print("ROI shape:", roi.shape)

#a plot i can actually see
plt.figure(figsize=(8,8))

plt.imshow(
    roi,
    cmap="gray",
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    vmin=np.percentile(image, 5),
    vmax=np.percentile(image, 99)
)

plt.colorbar(label="Pixel Value")

plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Zoomed Region of Interest")

plt.show()