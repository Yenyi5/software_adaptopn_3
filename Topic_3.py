'''
3) Toponym Placement within Lake Polygons
Description:    This script processes lake polygons from an OSM dataset, rasterizes and skeletonizes them, 
                and prepares and places templates for future toponym placement along the skeletons.
Authors:        Yenyi Wu, Sarah Kolodzie
Deadline:       31.07.2025

Python version: 3.11.0
'''

# pip install geopandas --> in terminal
# pip install matplotlib --> in terminal
# pip install rasterio --> in terminal
# pip install scikit-image --> in terminal
# pip install pygeoops --> in terminal
# pip install centerline --> in terminal
# pip install opencv-python --> in terminal
# pip install plantcv jupyterlab ipympl --> in terminal


'''
imports
'''
import geopandas as gpd
#import pygeoops
#from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
import skimage as ski
#from skimage.measure import find_contours
from plantcv import plantcv as pcv
from scipy.ndimage import rotate, gaussian_filter
from scipy.interpolate import splrep, splev, splprep


'''
functions
'''
# function to rasterize polygons
### source for transformation: ChatGPT
### source: https://pygis.io/docs/e_raster_rasterize.html
def rasterize_polygon(polygon, out_shape=(1000, 1000)): 
    '''
    Rasterizes a given polygon.
    
    Parameters: polygon --> polygon to rasterize
                out_shape --> shape of the output raster
    
    Returns:    rasterized polygon
    '''
    transform = from_origin(
        polygon.total_bounds[0],  # minx
        polygon.total_bounds[3],  # maxy
        (polygon.total_bounds[2] - polygon.total_bounds[0]) / out_shape[1],  # pixel width
        (polygon.total_bounds[3] - polygon.total_bounds[1]) / out_shape[0]   # pixel height
    ) 
    
    return rasterize(polygon.geometry, out_shape=out_shape, transform=transform)
    

# function to skeletonize a raster
### source: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
def skeletonize_raster(raster):
    '''
    Skeletonizes a given raster.
    
    Parameters: raster --> (binary) raster to skeletonize
    
    Returns:    skeletonized raster
    '''
    return ski.morphology.skeletonize(raster)
    

# function to segment skeletons using plantcv
### source: https://plantcv.readthedocs.io/en/stable/segment_skeleton/
def new_segments_of_skeletons(skel_img, mask_img):
    '''
    Segments the skeleton image using PlantCV.
    
    Parameters: skel_img --> skeleton image to segment
                mask_img --> mask image to use for segmentation
    
    Returns:    segmented image and objects (= segments of the skeleton)
    '''
    # plantcv expects uint8 images, so convert the skeleton and raster to uint8
    # source: https://gis.stackexchange.com/questions/481466/converting-image-data-type-from-uint16-to-uint8-using-python
    skel_img_uint8 = (skel_img * 255).astype(np.uint8)
    mask_img_uint8 = (mask_img * 255).astype(np.uint8)
    
    # segment the skeleton using plantcv
    segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skel_img_uint8, 
                                                        mask=mask_img_uint8)
    
    return segmented_img, obj


'''
load data
'''
# path to water shapefile from osm dataset Saxony
water_path = "C:\\Software_Adaption_Material\\water_osm\\water_osm\\gis_osm_water_a_free_1.shp" # polygons

# load dataframes --> chosen water bc the data is already polygons
water = gpd.read_file(water_path)

# check and if necessary change crs to UTM e.g. 32632
print(water.crs) # EPSG: 4326 (WGS84)
water = water.to_crs(epsg=25832)
print(water.crs) # EPSG: 25832 (UTM Zone 32N)

# plot to visualize
water.plot()
plt.title("water saxony")              
plt.show()

# to find attribute names and objects in the dataframe to get overview of data
print(water.columns)
print(water.head())


'''
polygons selected using QGIS --> osm_id used here as identifier
'''
# select polygons by osm_id
easy = water[water["osm_id"] == "9727383"] # .copy()
medium = water[water["osm_id"] == "6581385"]
hard = water[water["osm_id"] == "507087093"]
super_hard = water[water["osm_id"] == "1040656052"]

# plot the polygons together in a 2x2 format
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].set_title("easy")
easy.plot(ax=axes[0, 0])
axes[0, 1].set_title("medium")
medium.plot(ax=axes[0, 1])
axes[1, 0].set_title("hard")
hard.plot(ax=axes[1, 0])
axes[1, 1].set_title("super_hard")
super_hard.plot(ax=axes[1, 1])
plt.tight_layout()
plt.show()


'''
rasterize the polygons
# source: mainly used ChatGPT to help
'''
# rasterize the polygons
easy_raster = rasterize_polygon(easy, out_shape=(1000, 1000))
medium_raster = rasterize_polygon(medium, out_shape=(1000, 1000))
hard_raster = rasterize_polygon(hard, out_shape=(1000, 1000))
super_hard_raster = rasterize_polygon(super_hard, out_shape=(1000, 1000))

# plot the rasterized polygons together in a 2x2 format
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(easy_raster, cmap='gray')
axes[0, 0].set_title("Rasterized Easy Polygon")
axes[0, 1].imshow(medium_raster, cmap='gray')
axes[0, 1].set_title("Rasterized Medium Polygon")
axes[1, 0].imshow(hard_raster, cmap='gray')
axes[1, 0].set_title("Rasterized Hard Polygon")
axes[1, 1].imshow(super_hard_raster, cmap='gray')
axes[1, 1].set_title("Rasterized Super Hard Polygon")
plt.tight_layout()
plt.show()


'''
skeletonize using skimage
source: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
'''
# skeletonize the rasters
easy_skeleton = skeletonize_raster(easy_raster)
medium_skeleton = skeletonize_raster(medium_raster)
hard_skeleton = skeletonize_raster(hard_raster)
super_hard_skeleton = skeletonize_raster(super_hard_raster)

# plot the skeletons together in a 2x2 format
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(easy_skeleton, cmap='gray')
axes[0, 0].set_title("Skeletonized Easy Raster")
axes[0, 1].imshow(medium_skeleton, cmap='gray')
axes[0, 1].set_title("Skeletonized Medium Raster")
axes[1, 0].imshow(hard_skeleton, cmap='gray')
axes[1, 0].set_title("Skeletonized Hard Raster")
axes[1, 1].imshow(super_hard_skeleton, cmap='gray')
axes[1, 1].set_title("Skeletonized Super Hard Raster")
plt.tight_layout()
plt.show()


'''
segmentation of the skeletons using plantcv
# source: https://plantcv.readthedocs.io/en/stable/segment_skeleton/
'''
# segment the skeletons using the library PlantCV
easy_segmented_img, easy_obj = new_segments_of_skeletons(easy_skeleton, easy_raster)
medium_segmented_img, medium_obj = new_segments_of_skeletons(medium_skeleton, medium_raster)
hard_segmented_img, hard_obj = new_segments_of_skeletons(hard_skeleton, hard_raster)
super_hard_segmented_img, super_hard_obj = new_segments_of_skeletons(super_hard_skeleton, super_hard_raster)

# plot the segmented images together in a 2x2 format
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(easy_segmented_img, cmap='gray')
axes[0, 0].set_title("Segmented Easy Image")
axes[0, 1].imshow(medium_segmented_img, cmap='gray')
axes[0, 1].set_title("Segmented Medium Image")
axes[1, 0].imshow(hard_segmented_img, cmap='gray')
axes[1, 0].set_title("Segmented Hard Image")
axes[1, 1].imshow(super_hard_segmented_img, cmap='gray')
axes[1, 1].set_title("Segmented Super Hard Image")
plt.tight_layout()
plt.show()


# plot the segmented objects iteratively to check
for i in range(len(easy_obj)):
    x_coords = easy_obj[i][:, 0, 0]
    y_coords = easy_obj[i][:, 0, 1]
    plt.plot(x_coords, y_coords, color='red')
    plt.title(f"Segmented Object {i+1}")
    plt.gca().invert_yaxis()  # Flip y-axis 
    plt.show()


'''
make the segments smoother 
--> evtl einfach weglassen, weil es tw. nicht viel bringt bzw. es 
noch schlimmer macht durch schlaufen etc.

aber smoothing wäre schon gut eigentlich, damit die Buchstaben später 
besser/einfacher platziert werden können
'''
from scipy.signal import savgol_filter
# smooth segment using savgol_filter
# source: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-for-a-dataset
def smooth_segment(segment, window_length=15, polyorder=2):
    '''
    Smooths a segment using Savitzky-Golay filter.
    
    Parameters: segment --> segment to smooth
                window_length --> length of the smoothing window (must be odd)
                polyorder --> order of the polynomial used to fit the samples
    
    Returns:    smoothed segment
    '''
    if len(segment) < window_length:
        return segment  # Not enough points to smooth
    smoothed = savgol_filter(segment, window_length, polyorder, axis=0)
    return smoothed

# Smooth the segments using the function
easy_smoothed_segments = [smooth_segment(seg[:, 0, :]) for seg in easy_obj]
medium_smoothed_segments = [smooth_segment(seg[:, 0, :]) for seg in medium_obj]
hard_smoothed_segments = [smooth_segment(seg[:, 0, :]) for seg in hard_obj]
super_hard_smoothed_segments = [smooth_segment(seg[:, 0, :]) for seg in super_hard_obj]


# make function to plot original and smoothed segments in comparison
def plot_segments_comparison(original_segments, smoothed_segments, title_prefix="Segment"):
    '''
    Plots original and smoothed segments in comparison.
    
    Parameters: original_segments --> list of original segments
                smoothed_segments --> list of smoothed segments
                title_prefix --> prefix for the plot titles
    '''
    for i, (original, smoothed) in enumerate(zip(original_segments, smoothed_segments)):
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.plot(original[:, 0, 0], original[:, 0, 1], color='red')
        plt.title(f'{title_prefix} {i+1} - Original')
        plt.gca().invert_yaxis()  # Flip y-axis
        plt.subplot(1, 2, 2)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='blue')
        plt.title(f'{title_prefix} {i+1} - Smoothed')
        plt.gca().invert_yaxis()  # Flip y-axis
        plt.show()


# Plot the segments for each difficulty level
plot_segments_comparison(easy_obj, easy_smoothed_segments, title_prefix="Easy Segment")
plot_segments_comparison(medium_obj, medium_smoothed_segments, title_prefix="Medium Segment")
plot_segments_comparison(hard_obj, hard_smoothed_segments, title_prefix="Hard Segment")
plot_segments_comparison(super_hard_obj, super_hard_smoothed_segments, title_prefix="Super Hard Segment")



#####################################
# latest update: bis zum smoothing, das funktioniert aber noch nicht
# so richtig!

# nächste Schritte: 
# - Punkte entlang der Segmente verteilen
# - Orientierung der Segmente berechnen
# - Template für Buchstaben erstellen
# - Templates platzieren
#####################################




'''
calculate orientation of the separate segments? 
or rather, calculate orientation of the segment at specific points?
because we only need orientation of where the template will go to rotate it accordingly
'''





'''
make template for letters
# source: mainly used ChatGPT and Copilot
'''
# base template
letter_template = np.ones((50, 40), dtype=np.uint8)

# angles from -90 to +90 in 15° steps
angles = np.arange(-90, 91, 15)
# make rotated templates
rotated_templates = {}
for angle in angles:
    rotated = rotate(letter_template, angle, reshape=True, order=0)
    rotated_templates[angle] = (rotated > 0.5).astype(np.uint8)  # Ensure binary

# Pad all rotated templates to the same size
max_shape = np.array([t.shape for t in rotated_templates.values()]).max(axis=0)
for angle in angles:
    t = rotated_templates[angle]
    pad_height = max_shape[0] - t.shape[0]
    pad_width = max_shape[1] - t.shape[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    rotated_templates[angle] = np.pad(t, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

# Example: show all rotated templates
fig, axes = plt.subplots(1, 13, figsize=(14, 6))
for ax, angle in zip(axes.flat, angles):
    ax.imshow(rotated_templates[angle], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{angle}°")
plt.tight_layout()
plt.show()










