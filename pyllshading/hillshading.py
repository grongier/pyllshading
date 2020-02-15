"""hillshading"""

# The MIT License (MIT)
# Copyright (c) 2019 Guillaume Rongier
#
# Author: Guillaume Rongier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import math
import numpy as np
from numba import jit, prange
import scipy.ndimage as ndi
from rasterio.warp import Resampling
from matplotlib import cm, colors

from pyllshading.geomorphometry import _compute_gradient_value, _compute_horn_gradient_value, compute_horn_slope, compute_horn_aspect, compute_plan_curvature, compute_positive_max_curvature, compute_negative_min_curvature
from pyllshading.utils import mask_circular, resample, scale_minmax, scale_threshold, scale_posneg
from pyllshading.cmap import blue_yellow
from pyllshading.blending import composite_alpha


################################################################################
# Hillshading using Horn's slope and aspect


@jit(nopython=True)
def _compute_instensity_value(dx, dy, zenith, azimuth):
    '''
    Computes the instensity value.
    '''
    slope = math.atan(math.sqrt(dx*dx + dy*dy))
    aspect = math.atan2(-dx, dy)
    if aspect < 0.:
        aspect += 2*math.pi
        
    instensity = math.cos(zenith)*math.cos(slope) \
                 + math.sin(zenith)*math.sin(slope)*math.cos(azimuth - aspect)
    
    if instensity < 0.:
        return 0.
    return instensity
           

@jit(nopython=True, nogil=True, parallel=True)
def _hillshade_horn(elevation, spacing_y, spacing_x, azimuth, altitude, vert_exag):
    '''
    Computes the hillshade on a raster using Horn's slope and aspect.
    '''
    zenith = np.pi/2. - altitude*np.pi/180.
    azimuth *= np.pi/180.

    instensity = np.zeros(elevation.shape)

    # Interior
    for j in prange(1, elevation.shape[0] - 1):
        for i in range(1, elevation.shape[1] - 1):
            dy, dx = _compute_horn_gradient_value(elevation,
                                                  2*spacing_y, 2*spacing_x,
                                                  j + 1, j, j - 1,
                                                  i + 1, i, i - 1,
                                                  vert_exag)
            instensity[j, i] = _compute_instensity_value(dx, dy, zenith, azimuth)
            
    # Borders and corners
    for j in (0, elevation.shape[0] - 1):
        jp1 = j + 1 if j < elevation.shape[0] - 1 else elevation.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, elevation.shape[1]):
            ip1 = i + 1 if i < elevation.shape[1] - 1 else elevation.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            factor = 2 if i != 0 and i != elevation.shape[1] - 1 else 1
            dy, dx = _compute_gradient_value(elevation,
                                             spacing_y, factor*spacing_x,
                                             jp1, j, jm1,
                                             ip1, i, im1,
                                             vert_exag)
            instensity[j, i] = _compute_instensity_value(dx, dy, zenith, azimuth)
    for j in range(1, elevation.shape[0] - 1):
        for i in (0, elevation.shape[1] - 1):
            ip1 = i + 1 if i < elevation.shape[1] - 1 else elevation.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            dy, dx = _compute_gradient_value(elevation,
                                             2*spacing_y, spacing_x,
                                             j + 1, j, j - 1,
                                             ip1, i, im1,
                                             vert_exag)
            instensity[j, i] = _compute_instensity_value(dx, dy, zenith, azimuth)

    return instensity


def hillshade_horn(elevation,
                   spacing=1,
                   azimuth=315,
                   altitude=45,
                   vert_exag=1):
    '''
    Computes the hillshade on a raster using Horn's slope and aspect.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    azimuth : float (default 315)
        Angular direction of the Sun, measured clockwise from the North in degree
        from 0 to 360.

    altitude : float (default 45)
        Angle of illumination from the horizon in degree from 0 (horizontal) to
        90 (vertical).

    vert_exag : float (default 1)
        Vertical exaggeration.
        
    Returns
    -------
    
    intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.
    
    References
    ----------
    
    Horn, B. K. (1981).
    Hill shading and the reflectance map.
    Proceedings of the IEEE, 69(1), 14-47. https://doi.org/10.1109/PROC.1981.11918
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _hillshade_horn(elevation,
                           spacing[0], spacing[1],
                           azimuth, altitude, vert_exag)

################################################################################
# Hillshade postprocessing


def brighten_flat_areas(instensity,
                        elevation,
                        spacing=1,
                        flat_tone=0.95,
                        slope_threshold=2,
                        curvature=0.8):
    '''
    Brightens the flat areas of the intensity.
    
    Parameters
    ----------
    
    instensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.

    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    flat_tone : float (default 0.95)
        Intensity attributed to the flat areas, from 0 (dark) to 1 (bright).

    slope_threshold : float (default 2)
        Slope threshold below which an area is considered flat.

    curvature : float (default 0.8)
        Weight determining the influence of the brightening, from 0 (high 
        influence) to 1 (low influence).
        
    Returns
    -------
    
    intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.
    
    References
    ----------

    Jenny, B. (2001).
    An interactive approach to analytical relief shading.
    Cartographica: The International Journal for Geographic Information and Geovisualization, 38(1-2), 67-75.
    http://portal.survey.ntua.gr/main/courses/geoinfo/admcarto/lecture_notes/hill_shading/bibliography/jenny_2001.pdf
    
    '''
    slope = compute_horn_slope(elevation, spacing=spacing)*180/np.pi
    weight = np.zeros(instensity.shape)
    weight[slope > slope_threshold] = np.exp(np.log((slope[slope > slope_threshold]
                                                     - slope_threshold)/(90 - slope_threshold))*curvature)
    
    return weight*instensity + (1 - weight)*flat_tone


def add_atmospheric_perspective(intensity, elevation, C1=1.5, C2=0.25, altitude=45):
    '''
    Adds atmospheric perspective effects to the intensity by increasing the
    tonal contrast of the upper elevations and reducing that of the lower
    elevations.
    
    Parameters
    ----------
    
    intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.

    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.
        
    C1 : float (default 1.5)
        Factor equal to or larger than 1 defining the extent of the variation
        in contrast.

    C2 : float (default 0.25)
        Factor between -1 and 1 defining the extent of obscuring (when between 0
        excluded and 1 included) or clearing (when between -1 included and 0
        excluded)

    altitude : float (default 45)
        Angle of illumination from the horizon in degree from 0 (horizontal) to
        90 (vertical).
        
    Returns
    -------
    
    atm_intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.
    
    References
    ----------

    Brassel, K. (1974).
    A Model for Automatic Hill-Shading.
    The American Cartographer, 1:1, 15-27.
    https://www.tandfonline.com/doi/abs/10.1559/152304074784107818
    
    '''
    m = (np.nanmax(elevation) + np.nanmin(elevation))/2
    s = (np.nanmax(elevation) - np.nanmin(elevation))/2
    z_star = (elevation - m)/s

    zenith = np.pi/2. - altitude*np.pi/180.
    flat_intensity = np.cos(zenith)

    atm_intensity = (intensity - flat_intensity)*np.exp(z_star*np.log(C1)) + flat_intensity
    atm_intensity += C2*(z_star - 1)/2
    
    return atm_intensity

################################################################################
# MDOW hillshading


def generalize_elevation_mdow(elevation,
                              transform,
                              crs,
                              radius=1000,
                              generalize_spacing=1000):
    '''
    Generalizes the elevation for multidirectional, oblique-weighted (MDOW)
    hillshading.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.

    transform : affine.Affine()
        Affine transformation of the raster (see rasterio).

    crs : CRS or dict
        Coordinate reference system of the raster (see rasterio). 
        
    radius : int (default 1000)
        Radius of the circular window to filter the input raster, in the input
        raster spatial units.

    generalize_spacing : float (default 1000)
        Spacing between the cells of the generalized raster, in the input raster
        spatial units
        
    Returns
    -------
    
    generalized_elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of generalized elevation values.
    
    References
    ----------

    Mark, R. K. (1992).
    A multidirectional, oblique-weighted, shaded-relief image of the Island of Hawaii
    (No. 92-422). US Dept. of the Interior, US Geological Survey.
    https://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf
    
    '''
    if isinstance(generalize_spacing, (float, int)) == True:
        generalize_spacing = (generalize_spacing, generalize_spacing)

    generalized_elevation = None
    generalized_transform = None
    if generalize_spacing is not None:
        generalized_elevation, generalized_transform = resample(elevation,
                                                                transform,
                                                                crs,
                                                                generalize_spacing,
                                                                Resampling.cubic_spline)
    else:
        generalized_elevation = np.copy(elevation)
        generalize_spacing = (abs(transform[4]), transform[0])

    generalized_size = (int(np.ceil(radius/generalize_spacing[0])),
                        int(np.ceil(radius/generalize_spacing[1])))
    footprint = mask_circular(generalized_size)
    generalized_elevation = ndi.filters.generic_filter(generalized_elevation,
                                                       np.mean,
                                                       footprint=footprint)
    for i in range(2):
        generalized_elevation = ndi.filters.generic_filter(generalized_elevation,
                                                           np.mean,
                                                           footprint=footprint)

    if generalized_transform is not None:
        generalized_elevation, _ = resample(generalized_elevation,
                                            generalized_transform,
                                            crs,
                                            (transform[0], abs(transform[4])),
                                            Resampling.cubic_spline)

    return generalized_elevation


def hillshade_mdow(elevation,
                   transform,
                   crs,
                   radius=1000,
                   generalize_spacing=1000,
                   azimuths=None,
                   altitude=30,
                   vert_exag=1):
    '''
    Computes the hillshade on a raster using the multidirectional, oblique-
    weighted (MDOW) hillshading.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.

    transform : affine.Affine()
        Affine transformation of the raster (see rasterio).

    crs : CRS or dict
        Coordinate reference system of the raster (see rasterio). 
        
    radius : int (default 1000)
        Radius of the circular window to filter the input raster during
        generalization, in the input raster spatial units.

    generalize_spacing : float (default 1000)
        Spacing between the cells of the generalized raster, in the input raster
        spatial units

    azimuths : array-like, optional (default None)
        Multiple angular directions of the Sun, measured clockwise from the 
        North in degree from 0 to 360. Defaults are 225, 270, 315, and 360
        degrees.

    altitude : float (default 30)
        Angle of illumination from the horizon in degree from 0 (horizontal) to
        90 (vertical).

    vert_exag : float (default 1)
        Vertical exaggeration.
        
    Returns
    -------
    
    mdow_intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.
    
    References
    ----------

    Mark, R. K. (1992).
    A multidirectional, oblique-weighted, shaded-relief image of the Island of Hawaii
    (No. 92-422). US Dept. of the Interior, US Geological Survey.
    https://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf
    
    '''
    if azimuths is None:
        azimuths = np.linspace(225, 360, 4)
    
    generalized_array = generalize_elevation_mdow(elevation,
                                                  transform,
                                                  crs,
                                                  radius=radius,
                                                  generalize_spacing=generalize_spacing)
    generalized_aspect = compute_horn_aspect(vert_exag*generalized_array,
                                             spacing=(abs(transform[4]), transform[0]))
    
    mdow_intensity = np.zeros(elevation.shape)
    for i, azimuth in enumerate(azimuths):

        intensity = hillshade_horn(elevation,
                                   spacing=(abs(transform[4]), transform[0]),
                                   azimuth=azimuth,
                                   altitude=altitude,
                                   vert_exag=vert_exag)
        weights = np.sin(generalized_aspect - azimuth*np.pi/180)**2
        mdow_intensity += weights*intensity

    return mdow_intensity/2

################################################################################
# Swiss-style hillshading


def hillshade_swiss_style(elevation,
                          transform,
                          crs,
                          azimuth=315,
                          altitude=45,
                          vert_exag=1,
                          cmap_aerial=cm.gray,
                          cmap_smooth=blue_yellow,
                          alpha_smooth=0.35,
                          cmap_dem=cm.gist_earth,
                          alpha_dem=0.55):
    '''
    Computes the colored hillshade on a raster in Swiss style.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.

    transform : affine.Affine()
        Affine transformation of the raster (see rasterio).

    crs : CRS or dict
        Coordinate reference system of the raster (see rasterio). 

    azimuths : array-like, optional (default None)
        Multiple angular directions of the Sun, measured clockwise from the 
        North in degree from 0 to 360. Defaults are 225, 270, 315, and 360
        degrees.

    altitude : float (default 30)
        Angle of illumination from the horizon in degree from 0 (horizontal) to
        90 (vertical).

    vert_exag : float (default 1)
        Vertical exaggeration.

    cmap_aerial : Colormap (default cm.gray)
        Colormap for the aerial perspective.

    cmap_smooth : Colormap (default blue_yellow)
        Colormap for the smoothed hillshade (elevation mist).

    alpha_smooth : float (default 0.35)
        Weight for the smoothed hillshade.

    cmap_dem : Colormap (default cm.gist_earth)
        Colormap for the elevation.

    alpha_dem : float (default 0.55)
        Weight for the elevation.
        
    Returns
    -------
    
    rgba : ndarray, shape (n_cells_y, n_cells_x)
        Raster of color values.
    
    References
    ----------

    Barnes, D. (2002).
    Using ArcMap to Enhance Topographic Presentation.
    Cartographic Perspectives, (42), 5-11. https://doi.org/10.14714/CP42.549
    https://cartographicperspectives.org/index.php/journal/article/view/cp42-barnes
    
    '''
    intensity = hillshade_horn(elevation,
                               spacing=(abs(transform[4]), transform[0]),
                               azimuth=azimuth,
                               altitude=altitude,
                               vert_exag=vert_exag)
    aerial_perspective = elevation/5 + intensity
    norm = colors.Normalize()
    aerial_perspective = cmap_aerial(norm(np.ma.masked_invalid(aerial_perspective)))
    
    footprint = mask_circular(4)
    smooth_intensity = ndi.filters.median_filter(intensity, footprint=footprint)
    smooth_intensity = cmap_smooth(np.ma.masked_invalid(smooth_intensity),
                                   alpha=alpha_smooth)
    
    norm = colors.Normalize()
    dem = cmap_dem(norm(np.ma.masked_invalid(elevation)), alpha=alpha_dem)
    
    rgba = composite_alpha(smooth_intensity, aerial_perspective)
    rgba = composite_alpha(dem, rgba)
    
    return rgba

################################################################################
# Terrain sculptor


def _exaggerate_ridges(elevation,
                       spacing,
                       nb_filtering_elevation,
                       ridges_plancurvature_weight,
                       nb_filtering_weight,
                       ridges_exaggeration):
    '''
    Exaggerates the ridges.
    '''
    elevation_exaggerated = ndi.gaussian_filter(elevation,
                                                0.7*nb_filtering_elevation)

    plan_curvature = compute_plan_curvature(elevation_exaggerated,
                                            spacing=spacing)
    plan_curvature = scale_posneg(plan_curvature, -ridges_plancurvature_weight)

    max_curvature = compute_positive_max_curvature(elevation_exaggerated,
                                                   spacing=spacing)
    max_curvature = scale_minmax(max_curvature)

    concavity_weight = plan_curvature + max_curvature
    concavity_weight = ndi.gaussian_filter(concavity_weight,
                                           0.7*nb_filtering_weight)
    concavity_weight = scale_minmax(concavity_weight)

    return elevation_exaggerated*(concavity_weight*(ridges_exaggeration - 1) + 1)


def _exaggerate_valleys(elevation,
                        spacing,
                        nb_filtering_elevation,
                        max_valleys_curvature,
                        valleys_exaggeration):
    '''
    Exaggerates the valleys.
    '''
    elevation_deepened = ndi.gaussian_filter(elevation,
                                             0.7*nb_filtering_elevation)

    min_curvature = compute_negative_min_curvature(elevation_deepened,
                                                   spacing=spacing)
    convexity_weight = scale_threshold(min_curvature,
                                       max_valleys_curvature*np.nanmax(min_curvature))

    return elevation_deepened*(convexity_weight*(valleys_exaggeration - 1) + 1)


def _compute_slope_weight(elevation,
                          spacing,
                          nb_filtering_elevation,
                          slope_threshold):
    '''
    Computes the slope weight to fuse exaggerated ridges and valleys.
    '''
    slope = compute_horn_slope(elevation, spacing=spacing)
    slope = ndi.gaussian_filter(slope, 0.4*nb_filtering_elevation)

    slope_weight = scale_threshold(slope, slope_threshold*np.pi/180)
    slope_weight = ndi.gaussian_filter(slope_weight, 0.4*nb_filtering_elevation)

    return slope_weight


def generalize_elevation_terrain_sculptor(elevation,
                                          spacing,
                                          nb_filtering_elevation=10,
                                          nb_filtering_ridges=5,
                                          ridges_plancurvature_weight=1.5,
                                          nb_filtering_ridges_weight=1,
                                          ridges_exaggeration=1.25,
                                          nb_filtering_valleys=5,
                                          max_valleys_curvature=0.5,
                                          valleys_exaggeration=0.4,
                                          slope_threshold=15):
    '''
    Generalizes the elevation for terrain sculptor.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    nb_filtering_elevation : int (default 10)
        Number of elevation filtering when exaggerating ridges and valleys.

    nb_filtering_ridges : int (default 5)
        Number of filtering of the ridges.

    ridges_plancurvature_weight : int (default 1.5)
        Weight of the plan curvature when exaggerating ridges.

    nb_filtering_ridges_weight : int (default 1)
        Number of filtering for the concavity weight when exaggerating ridges.

    ridges_exaggeration : int (default 1.25)
        Exaggeration of the ridges.

    nb_filtering_valleys : int (default 5)
        Number of filtering of the valleys.

    max_valleys_curvature : int (default 0.5)
        Maximum curvature of the valleys.

    valleys_exaggeration : int (default 0.4)
        Exaggeration of the valleys.

    slope_threshold : int (default 15)
        Number of elevation filtering when exaggerating ridges and valleys.
        
    Returns
    -------
    
    generalized_elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of generalized elevation values.
    
    References
    ----------
    
    Leonowicz, A. M., Jenny, B., & Hurni, L. (2010).
    Terrain Sculptor: Generalizing Terrain Models for Relief Shading.
    Cartographic Perspectives, (67), 51-60.
    https://doi.org/10.14714/CP67.114

    Jenny, B. (2011).
    Terrain Sculptor 2.0.
    https://github.com/OSUCartography/TerrainSculptor
    
    '''
    ridges = _exaggerate_ridges(elevation,
                                spacing,
                                nb_filtering_ridges,
                                ridges_plancurvature_weight,
                                nb_filtering_ridges_weight,
                                ridges_exaggeration)
    valleys = _exaggerate_valleys(elevation,
                                  spacing,
                                  nb_filtering_valleys,
                                  max_valleys_curvature,
                                  valleys_exaggeration)
    slope_weight = _compute_slope_weight(elevation,
                                         spacing,
                                         nb_filtering_elevation,
                                         slope_threshold)
    
    return slope_weight*ridges + (1 - slope_weight)*valleys


def hillshade_terrain_sculptor(elevation,
                               spacing=1,
                               nb_filtering_elevation=10,
                               nb_filtering_ridges=5,
                               ridges_plancurvature_weight=1.5,
                               nb_filtering_ridges_weight=1,
                               ridges_exaggeration=1.25,
                               nb_filtering_valleys=5,
                               max_valleys_curvature=0.5,
                               valleys_exaggeration=0.4,
                               slope_threshold=15,
                               azimuth=315,
                               altitude=45,
                               vert_exag=1):
    '''
    Computes the hillshade on a raster using terrain sculptor.
    
    Parameters
    ----------
    
    elevation : ndarray, shape (n_cells_y, n_cells_x)
        Raster of elevation values.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    nb_filtering_elevation : int (default 10)
        Number of elevation filtering when exaggerating ridges and valleys.

    nb_filtering_ridges : int (default 5)
        Number of filtering of the ridges.

    ridges_plancurvature_weight : int (default 1.5)
        Weight of the plan curvature when exaggerating ridges.

    nb_filtering_ridges_weight : int (default 1)
        Number of filtering for the concavity weight when exaggerating ridges.

    ridges_exaggeration : int (default 1.25)
        Exaggeration of the ridges.

    nb_filtering_valleys : int (default 5)
        Number of filtering of the valleys.

    max_valleys_curvature : int (default 0.5)
        Maximum curvature of the valleys.

    valleys_exaggeration : int (default 0.4)
        Exaggeration of the valleys.

    slope_threshold : int (default 15)
        Number of elevation filtering when exaggerating ridges and valleys.

    azimuth : float (default 315)
        Angular direction of the Sun, measured clockwise from the North in degree
        from 0 to 360.

    altitude : float (default 45)
        Angle of illumination from the horizon in degree from 0 (horizontal) to
        90 (vertical).

    vert_exag : float (default 1)
        Vertical exaggeration.
        
    Returns
    -------
    
    intensity : ndarray, shape (n_cells_y, n_cells_x)
        Raster of illumination values between 0-1, where 0 is completely dark 
        and 1 is completely illuminated.
    
    References
    ----------
    
    Leonowicz, A. M., Jenny, B., & Hurni, L. (2010).
    Terrain Sculptor: Generalizing Terrain Models for Relief Shading.
    Cartographic Perspectives, (67), 51-60.
    https://doi.org/10.14714/CP67.114

    Jenny, B. (2011).
    Terrain Sculptor 2.0.
    https://github.com/OSUCartography/TerrainSculptor
    
    '''
    elevation_generalized = generalize_elevation_terrain_sculptor(elevation,
                                                                  spacing,
                                                                  nb_filtering_elevation,
                                                                  nb_filtering_ridges,
                                                                  ridges_plancurvature_weight,
                                                                  nb_filtering_ridges_weight,
                                                                  ridges_exaggeration,
                                                                  nb_filtering_valleys,
                                                                  max_valleys_curvature,
                                                                  valleys_exaggeration,
                                                                  slope_threshold)
    
    return hillshade_horn(elevation_generalized,
                          spacing=spacing,
                          azimuth=azimuth,
                          altitude=altitude,
                          vert_exag=vert_exag)
