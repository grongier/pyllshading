"""Utils"""

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
from rasterio.warp import reproject, Resampling


################################################################################
# Circular mask


def mask_circular(radius):
    '''
    Creates a circular mask.
    
    Parameters
    ----------
    
    radius : int or float
        Radius of the mask in cell.
        
    Returns
    -------
    
    mask : ndarray, shape (2*radius, 2*radius)
        The mask, with 1 being inside and 0 outside.
        
    '''
    if isinstance(radius, (float, int)) == True:
        radius = (radius, radius)

    y, x = np.ogrid[-radius[0]:radius[0] + 1, -radius[1]:radius[1] + 1]

    return (x/radius[1])**2 + (y/radius[0])**2 <= 1

################################################################################
# Resampling


def resample(src_array,
             src_transform,
             src_crs,
             dst_spacing,
             resampling=Resampling.cubic):
    '''
    Resamples a raster at a different resolution.
    
    Parameters
    ----------
    
    src_array : ndarray, shape (n_cells_y, n_cells_x)
        Source raster to resample.

    src_transform : affine.Affine()
        Source affine transformation.

    src_crs : CRS or dict
        Source coordinate reference system. This system is preserved during
        resampling. 

    dst_spacing : int or Resampling (default Resampling.cubic)
        Cell spacing for the resampled raster (i.e., new resolution).

    resampling : int or float
        Resampling method to use.
        
    Returns
    -------
    
    dst_array : ndarray, shape (n_cells_y, n_cells_x)
        The resampled raster.

    dst_transform : affine.Affine()
        Affine transformation of the resampled raster.

    Examples
    --------

    # Resample an array
    dst_array, dst_transform = resample(src_array, src_transform, src_crs, dst_spacing)
    # Back-transform to the original resolution
    back_array, back_transform = resample(dst_array, dst_transform, src_crs, src_spacing)
        
    '''
    if isinstance(dst_spacing, (float, int)) == True:
        dst_spacing = (dst_spacing, dst_spacing)

    dst_height = int(src_array.shape[0]*abs(src_transform[4])/dst_spacing[1])
    dst_width = int(src_array.shape[1]*src_transform[0]/dst_spacing[0])

    dst_spacing_height = src_array.shape[0]*src_transform[4]/dst_height
    dst_spacing_width = src_array.shape[1]*src_transform[0]/dst_width

    dst_transform = (dst_spacing_width,
                     0,
                     src_transform[2],
                     0,
                     dst_spacing_height,
                     src_transform[5])

    dst_array = np.empty((dst_height, dst_width))
    reproject(src_array,
              dst_array,
              src_transform=src_transform,
              src_crs=src_crs,
              dst_transform=dst_transform,
              dst_crs=src_crs,
              dst_nodata=np.nan,
              resampling=resampling)

    return dst_array, dst_transform

################################################################################
# Scaling


def scale_minmax(a):
    '''
    Scale an array for [minimum, maximum] to [0, 1].
    
    Parameters
    ----------
    
    a : ndarray
        The array to scale.
        
    Returns
    -------
    
    s : ndarray
        The scaled array.
        
    '''
    min_a = np.nanmin(a)
    max_a = np.nanmax(a)
    
    if max_a == min_a:
        return np.ones(a.shape)
    return (a - min_a)/(max_a - min_a)


def scale_threshold(a, threshold):
    '''
    Scale an array based on a threshold.
    
    Parameters
    ----------
    
    a : ndarray
        The array to scale.

    threshold : float
        The threshold.
        
    Returns
    -------
    
    s : ndarray
        The scaled array.
        
    '''
    factor = 1/threshold
    s = np.ones(a.shape)
    s[a <= threshold] = a[a <= threshold]*factor

    return s
    

def scale_posneg(a, factor):
    '''
    Scale an array based on its positive and negative values.
    
    Parameters
    ----------
    
    a : ndarray
        The array to scale.

    factor : ndarray
        The factor to modify the scaled range.
        
    Returns
    -------
    
    s : ndarray
        The scaled array.
        
    '''
    positive_factor = factor/abs(np.nanmax(a))
    negative_factor = factor/abs(np.nanmin(a))

    s = positive_factor*a
    s[a < 0.] = negative_factor*a[a < 0.]
    
    return s
