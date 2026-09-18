# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:42:04 2026

@author: ekim
"""

#show on monday 
from astropy.io import fits 
import matplotlib.pyplot as plt 
import numpy as np 
import glob 
import os 
import pandas as pd 
 
 
folder = r"C:/Users/ekim/Documents/research/SN1&SN2/SN1" 
 
fits_files = sorted(glob.glob(os.path.join(folder, "*.fits"))) 
 
print(f"Found {len(fits_files)} FITS files") 
 
if len(fits_files) == 0: 
    raise ValueError("No FITS files found. Check folder path.") 
 

x_min = 2287
x_max = 2297

y_min = 2443
y_max = 2451

 
scale = 0.00376      # mm/pixel 
 
all_profiles = [] 
all_x_centered = [] 
all_x_og = [] 
 
# LOOP THROUGH ALL IMAGES 
for file in fits_files:  
    #opens all fits files  
    with fits.open(file) as hdul:  
        image = hdul[0].data.astype(float)  
  
    # Crop ROI  
    roi = image[y_min:y_max, x_min:x_max]  
  
    if roi.size == 0:  
        print("Empty ROI:", file)  
        continue  
  
    # ==========================================================
    # FIND BRIGHTEST PIXEL IN ROI
    # ==========================================================
    
    brightest_index = np.unravel_index(np.argmax(roi), roi.shape)
    
    brightest_row = brightest_index[0]
    brightest_column = brightest_index[1]

    

    # ==========================================================
    # EXTRACT HORIZONTAL CROSS SECTION
    # ==========================================================
    
    profile = roi[brightest_row, :]
  
    # Normalize profile between 0 and 1  
    if profile.max() == profile.min():  
        print("Flat profile:", file)  
        continue  
  
    profile = (profile - profile.min()) / (profile.max() - profile.min())  
  
    # Save normalized profile  
    all_profiles.append(profile)
    
    
    # ========================================================== 
    # CENTER EACH INDIVIDUAL FITS FILE ON ITS WEIGHTED MEAN 
    # ========================================================== 
 
    #creates a pizel value for each intensity 
    # the fle doesnt alreadly inherantly have a pixel number store for each intensity, we just know that each intensity value is for each pixel
    x_pixels = np.arange(len(profile)) 
 
    # Calculate weighted mean position 
    weighted_mean = np.sum(x_pixels * profile) / np.sum(profile) 
 
    # Subtract weighted mean from every x-value 
    x_centered = ( 
        x_pixels - weighted_mean 
    ) * scale 
 
    # Save new centered x-values 
    all_x_centered.append(x_centered) 
 
print(
    f"{os.path.basename(file)}: "
    f"row = {brightest_row}, "
    f"column = {brightest_column}"
)

#stores all the x and y values of the fits files after normalized and centered
 
all_profiles = np.array(all_profiles) 
all_x_centered = np.array(all_x_centered) 
 
print("Profiles shape:", all_profiles.shape) 



# Plot the ROI and indicate the row being used
#reason it is 0-1 intensity scale is bc matplotlib intenally scales it to that just for displaying
plt.figure(figsize=(6, 6))

im = plt.imshow(
    roi,
    cmap="gray",
    origin="lower",
    vmin=image.min(),
    vmax=image.max()
)



plt.axhline(
    brightest_row,
    linewidth=2,
    label=f"Selected row = {brightest_row}"
)

plt.scatter(
    brightest_column,
    brightest_row,
    marker="+",
    s=150,
    linewidths=2
)

cbar = plt.colorbar(im)
cbar.set_label("Pixel Intensity")
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("ROI - Brightest Pixel and Selected Row")
plt.legend()

plt.show()
 
# ========================================================== 
# CREATE COMMON X-AXIS FOR CENTERED PROFILES 
# ========================================================== 
 
# Find the overlapping x-range shared by all centered profiles 

#max and min x valuesafter centered
x_min_common = np.max(all_x_centered[:, 0]) 
x_max_common = np.min(all_x_centered[:, -1]) 
 
# Create common x-axis 
#just to make it visually look bettter 

#linspace parameters(beginning, end, number of tick marks)
#(-1,1,5) = [-1.0, -0.5, 0.0, 0.5, 1.0]
x_common = np.linspace( 
    x_min_common, 
    x_max_common, 
    len(all_profiles[0]) 
) 
 
# Store centered profiles 
 
centered_profiles = [] 
 
#all fits files are in the same x-axis before averaging them 
 # now moves the new intensity values at the new x-values 

#goes through each intensity in all fits files
#x-common sets new tick marks so if there is no x= -0.5, it estimates the value here
for i in range(len(all_profiles)): 
    #interp estimates intensity between two points 
    centered_profile = np.interp( 
        x_common, #Where I want values
        all_x_centered[i],  #where my existing data is 
        all_profiles[i] #what the existing data values are 
    ) 
 
    centered_profiles.append(centered_profile) 
 
 
# Convert to NumPy array 
 
centered_profiles = np.array(centered_profiles) 
 
all_position = all_x_centered.flatten() 
 
all_intensity = all_profiles.flatten() 
 
print(centered_profiles.shape) 
 
 
# Calculate average AFTER every individual FITS 
 
# file has already been weighted-mean centered 
 
mean_profile = np.mean( 
    centered_profiles, 
    axis=0 
) 
 
print(mean_profile.shape) 
 
 
# ========================================================== 
# FWHM OF AVERAGE FITS PROFILE 
# ========================================================== 
 
# Half maximum 
half_max = 0.5

 
# Find where the average curve crosses 0.5 
 
above_half = mean_profile >= half_max 

#finds out where the curve goes from below to above the curve, stored as variable 
crossings = np.where( 
    np.diff(above_half.astype(int)) != 0 
)[0] 
 

# if there are two points where graph goes from below to above(crossing)

if len(crossings) >= 2: 
 
    # ====================================================== 
    # LEFT HALF-MAXIMUM CROSSING 
    # ====================================================== 
    # the first crossing 
    #there is no gurnatee that there is a x value at 0.5 intenity we take average of two nearest points to get FWHM
    i1 = crossings[0] 
 
    #x_common is the list of x-values on the same axis
    x_left_1 = x_common[i1] 
    x_left_2 = x_common[i1 + 1] 
 
    y_left_1 = mean_profile[i1] 
    y_left_2 = mean_profile[i1 + 1] 
 
    # Linear interpolation to find exact x position 
 
    x1 = x_left_1 + ( 
        (half_max - y_left_1) 
        / (y_left_2 - y_left_1) 
        * (x_left_2 - x_left_1) 
    ) 
 
 
    # ====================================================== 
    # RIGHT HALF-MAXIMUM CROSSING 
    # ====================================================== 
 
    i2 = crossings[-1] 
 
    x_right_1 = x_common[i2] 
    x_right_2 = x_common[i2 + 1] 
 
    y_right_1 = mean_profile[i2] 
    y_right_2 = mean_profile[i2 + 1] 
 
    # Linear interpolation to find exact x position 
 
    x2 = x_right_1 + ( 
        (half_max - y_right_1) 
        / (y_right_2 - y_right_1) 
        * (x_right_2 - x_right_1) 
    ) 
 
 
    # Calculate FWHM 
 
    fwhm = x2 - x1 
 
    print("====================================") 
    print("FWHM OF AVERAGE FITS PROFILE") 
    print("====================================") 
    print(f"Left half-maximum position:  {x1:.8f} mm") 
    print(f"Right half-maximum position: {x2:.8f} mm") 
    print(f"FWHM: {fwhm:.8f} mm") 
    print(f"FWHM: {fwhm * 1000:.8f} micrometers") 
 
else: 
 
    print("Could not find two half-maximum crossings.") 
 
 
# ========================================================== 
# SIMULATION!! 
# ========================================================== 
 
intensities = pd.read_csv( 
    r"C:/Users/ekim/Documents/research/metrocampics/0.2_intensities.csv", 
    header=None 
) 
 
# Convert to a NumPy array 
 
image = intensities.to_numpy() 
 
 
# Display image 
 
plt.figure(figsize=(6, 6)) 
 
plt.imshow( 
    image, 
    origin='lower', 
    extent=[ 
        -0.0195 / 2, 
        0.0195 / 2, 
        -0.0195 / 2, 
        0.0195 / 2 
    ], 
    cmap='gray' 
) 
 
plt.colorbar(label='Intensity') 
 
plt.xlabel('X Position (mm)') 
plt.ylabel('Y Position (mm)') 
 
plt.title('Intensity Map') 
 
plt.show() 
 
 
#Now making the cross section: 
 
#taking the cross section of only the middle row 
 
row = image.shape[0] // 2 
 
cross_section = image[row, :] 
 
 
# Create x-axis in physical units 
 
num_cols = image.shape[1] 
 
image_width = 0.0195  # mm 
 
x = np.linspace( 
    -image_width / 2, 
    image_width / 2, 
    num_cols 
) 
 
 
# Plot 
 
plt.figure(figsize=(8,4)) 
 
plt.plot( 
    x, 
    cross_section, 
    linewidth=2 
) 
 
plt.xlabel("X Position (mm)") 
 
plt.ylabel("Intensity") 
 
plt.title(f"Horizontal Cross Section (Row {row})") 
 
plt.grid(True) 
 
plt.show() 
 
 
# COMBINED GRAPH 
 
# Normalize CSV cross section between 0 and 1 
 
cross_section_normalized = ( 
    cross_section - cross_section.min() 
) / ( 
    cross_section.max() - cross_section.min() 
) 
 
 
# ========================================================== 
# FWHM OF ZEMAX SIMULATION 
# ========================================================== 
 
# Half maximum 
 
simulation_half_max = 0.5
 
# Find where the simulation curve crosses 0.5 
 
simulation_above_half = ( 
    cross_section_normalized >= simulation_half_max 
) 
 
simulation_crossings = np.where( 
    np.diff( 
        simulation_above_half.astype(int) 
    ) != 0 
)[0] 
 
 
if len(simulation_crossings) >= 2: 
 
    # ====================================================== 
    # LEFT HALF-MAXIMUM CROSSING 
    # ====================================================== 
 
    sim_i1 = simulation_crossings[0] 
 
    sim_x_left_1 = x[sim_i1] 
    sim_x_left_2 = x[sim_i1 + 1] 
 
    sim_y_left_1 = cross_section_normalized[sim_i1] 
    sim_y_left_2 = cross_section_normalized[sim_i1 + 1] 
 
    # Linear interpolation to find exact x position 
 
    sim_x1 = sim_x_left_1 + ( 
        (simulation_half_max - sim_y_left_1) 
        / (sim_y_left_2 - sim_y_left_1) 
        * (sim_x_left_2 - sim_x_left_1) 
    ) 
 
 
    # ====================================================== 
    # RIGHT HALF-MAXIMUM CROSSING 
    # ====================================================== 
 
    sim_i2 = simulation_crossings[-1] 
 
    sim_x_right_1 = x[sim_i2] 
    sim_x_right_2 = x[sim_i2 + 1] 
 
    sim_y_right_1 = cross_section_normalized[sim_i2] 
    sim_y_right_2 = cross_section_normalized[sim_i2 + 1] 
 
    # Linear interpolation to find exact x position 
 
    sim_x2 = sim_x_right_1 + ( 
        (simulation_half_max - sim_y_right_1) 
        / (sim_y_right_2 - sim_y_right_1) 
        * (sim_x_right_2 - sim_x_right_1) 
    ) 
 
 
    # Calculate FWHM 
 
    simulation_fwhm = sim_x2 - sim_x1 
 
    print("====================================") 
    print("FWHM OF ZEMAX SIMULATION") 
    print("====================================") 
    print(f"Left half-maximum position:  {sim_x1:.8f} mm") 
    print(f"Right half-maximum position: {sim_x2:.8f} mm") 
    print(f"FWHM: {simulation_fwhm:.8f} mm") 
    print(f"FWHM: {simulation_fwhm * 1000:.8f} micrometers") 
 
else: 
 
    print("Could not find two simulation half-maximum crossings.") 
 
 
# Plot CSV cross section on the same graph 
 
plt.figure(figsize=(10, 6)) 
 
 
# Plot all FITS profiles 






for i in range(len(centered_profiles)):

    plt.plot(
        x_common,
        centered_profiles[i],
        color="steelblue",
        alpha=0.25,
        linewidth=1,
        drawstyle="steps-mid"
    )


# Plot average FITS profile

plt.plot(
    x_common,
    mean_profile,
    color="black",
    linewidth=3,
    drawstyle="steps-mid",
    label="Real Photo Avg"
)
 
# plt.plot(
#     x_common,
#     mean_profile,
#     color="purple",
#     linewidth=3,
#     label="Real Photo Avg Curve"
# )
 
# Plot Top Hat 
 
edge = 0.1 * 0.049 
 
plt.plot( 
    [-0.02, -edge, -edge, edge, edge, 0.02], 
    [0, 0, 1, 1, 0, 0], 
    color="red", 
    linewidth=2, 
    label="Top Hat" 
) 
 
 
# Plot CSV cross section 
 
plt.plot( 
    x, 
    cross_section_normalized, 
    color="green", 
    linewidth=3, 
    label="Zemax Simulation" 
) 
 
 
# # ========================================================== 
# # FWHM ON GRAPH 
# # ========================================================== 
 
# if len(crossings) >= 2: 
 
#     # Plot the real photo half-maximum line 
 
#     plt.axhline( 
#         half_max, 
#         color="purple", 
#         linestyle="--", 
#         linewidth=1.5, 
#         label="Real Photo Half Maximum" 
#     ) 
 
#     # Plot vertical lines at real photo half-maximum positions 
 
#     plt.axvline( 
#         x1, 
#         color="purple", 
#         linestyle="--", 
#         linewidth=1.5 
#     ) 
 
#     plt.axvline( 
#         x2, 
#         color="purple", 
#         linestyle="--", 
#         linewidth=1.5 
#     ) 
 
#     # Mark the two real photo half-maximum points 
 
#     plt.scatter( 
#         [x1, x2], 
#         [half_max, half_max], 
#         color="purple", 
#         zorder=5 
#     ) 
 
#     # Label the real photo FWHM 
 
#     plt.text( 
#         (x1 + x2) / 2, 
#         half_max + 0.05, 
#         f"Real Photo FWHM = {fwhm * 1000:.3f} µm", 
#         ha="center", 
#         fontsize=11 
#     ) 
 
 
# # ========================================================== 
# # SIMULATION FWHM ON GRAPH 
# # ========================================================== 
 
# if len(simulation_crossings) >= 2: 
 
#     # Plot the simulation half-maximum line 
 
#     plt.axhline( 
#         simulation_half_max, 
#         color="orange", 
#         linestyle="--", 
#         linewidth=1.5, 
#         label="Simulation Half Maximum" 
#     ) 
 
#     # Plot vertical lines at simulation half-maximum positions 
 
#     plt.axvline( 
#         sim_x1, 
#         color="orange", 
#         linestyle="--", 
#         linewidth=1.5 
#     ) 
 
#     plt.axvline( 
#         sim_x2, 
#         color="orange", 
#         linestyle="--", 
#         linewidth=1.5 
#     ) 
 
#     # Mark the two simulation half-maximum points 
 
#     plt.scatter( 
#         [sim_x1, sim_x2], 
#         [simulation_half_max, simulation_half_max], 
#         color="orange", 
#         zorder=5 
#     ) 
 
#     # Label the simulation FWHM 
 
#     plt.text( 
#         (sim_x1 + sim_x2) / 2, 
#         simulation_half_max - 0.10, 
#         f"Simulation FWHM = {simulation_fwhm * 1000:.3f} µm", 
#         ha="center", 
#         fontsize=11 
#     ) 
 
 
# ========================================================== 
# GRAPH SETTINGS 
# ========================================================== 
 
plt.xlabel("X Position (mm)") 
 
plt.ylabel("Normalized Intensity") 
 
plt.title("Real Photo, Zemax Simulation, and Top Hat Cross Sections") 
 
plt.ylim(-0.05, 1.05) 
 
plt.grid(True) 
 
plt.legend() 
 
plt.tight_layout() 
 
plt.show() 
 
 
