"""Blending"""

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


import numpy as np
import skimage.exposure as exp


################################################################################
# NAGI fusion method


def _fuse_panchromatic(rgba, intensity):
    '''
    Adds a panchromatic image (intensity) to each band of a multispectral
    image (rgba).
    '''
    fuzed_rgba = np.copy(rgba)
    valid = np.logical_and(np.logical_not(np.isnan(rgba).any(axis=-1)),
                           np.logical_not(np.isnan(intensity)))
    fuzed_rgba[valid, :3] = 0.5*(rgba[valid, :3] + intensity[valid, np.newaxis])
    fuzed_rgba[np.logical_not(valid), 3] = 0.

    return fuzed_rgba


def blend_nagi(rgba, intensity, gamma=1.5, in_range=(10/255, 220/255)):
    '''
    No Alteration of Grayscale or Intensity (NAGI) fusion method.
    
    Parameters
    ----------
    
    rgba : ndarray, shape (n_cells_y, n_cells_x, 3)
        Red-Green-Blue-Alpha for each cell of the raster.
        
    intensity : ndarray, shape (n_cells_y, n_cells_x)
        Intensity for each cell of the raster (i.e., hillshade).
        
    gamma : float (default 1.5)
        Degree of contrast between the light and dark areas. A value lower than
        1 increases the contrast of the light areas at the expense of the dark
        areas; A value higher than 1 increases the contrast of the dark areas at
        the expense of the light areas.
        
    in_range : array-like (default (10/255, 220/255) )
        Minimum and maximum values to truncate each of the three input RGB bands.
        
    Returns
    -------
    
    blended_rgba : ndarray, shape (3, n_cells_y, n_cells_x)
        Red-Green-Blue-Alpha for each cell of the raster after fusion with the
        intensity.
    
    References
    ----------
    
    Nagi, R. S. (2012).
    Maintaining detail and color definition when integrating color and grayscale
    rasters using No Alteration of Grayscale or Intensity (NAGI) fusion method
    In proceedings of AutoCarto 2012, Columbus, Ohio, USA
    http://www.cartogis.org/docs/proceedings/2012/Nagi_AutoCarto2012.pdf

    Nagi, R. S. (2012).
    Combining colored and grayshade rasters with high fidelity
    ArcGIS Blog
    https://www.esri.com/arcgis-blog/products/product/imagery/combining-colored-and-grayshade-rasters-with-high-fidelity/?rmedium=redirect&rsource=blogs.esri.com/esri/arcgis/2012/01/18/combining-colored-and-grayshade-rasters-with-high-fidelity
    
    '''
    blended_rgba = _fuse_panchromatic(rgba, intensity)

    blended_rgba = exp.adjust_gamma(blended_rgba, gamma=gamma)
    blended_rgba = exp.rescale_intensity(blended_rgba,
                                         in_range=in_range,
                                         out_range=(0., 1.))

    return blended_rgba

################################################################################
# Alpha compositing


def composite_alpha(rgba_1, rgba_2):
    '''
    Alpha compositing between two RGBA rasters.
    
    Parameters
    ----------
    
    rgba_1 : ndarray, shape (n_cells_y, n_cells_x, 4)
        Red-Green-Blue-Alpha for each cell of the first raster.
        
    rgba_2 : ndarray, shape (n_cells_y, n_cells_x, 4)
        Red-Green-Blue-Alpha for each cell of the second raster.
        
    Returns
    -------
    
    composite : ndarray, shape ( n_cells_y, n_cells_x, 4)
        Red-Green-Blue-Alpha for each cell of the composite raster.

    '''
    composite = np.empty(rgba_1.shape)
    composite[..., 3] = rgba_1[..., 3] + rgba_2[..., 3]*(1 - rgba_1[..., 3])
    composite[..., :3] = rgba_1[..., 3:4]*rgba_1[..., :3] \
                         + rgba_2[..., 3:4]*rgba_2[..., :3]*(1 - rgba_1[..., 3:4])
    composite[..., :3][composite[..., 3] != 0] /= composite[..., 3:4][composite[..., 3] != 0]
    
    return composite
                                                       