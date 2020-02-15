"""Geomorphometry"""

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


################################################################################
# Horn's slope and aspect


@jit(nopython=True)
def _compute_gradient_value(array,
                            spacing_y, spacing_x,
                            jp1, j, jm1,
                            ip1, i, im1,
                            vert_exag=1):
    '''
    Computes the gradient value along a border.
    '''
    dy = vert_exag*(array[jp1, i] - array[jm1, i])/spacing_y
    dx = vert_exag*(array[j, ip1] - array[j, im1])/spacing_x

    return dy, dx


@jit(nopython=True)
def _compute_horn_gradient_value(array,
                                 spacing_y, spacing_x,
                                 jp1, j, jm1,
                                 ip1, i, im1,
                                 vert_exag=1):
    '''
    Computes Horn's gradient value.
    '''
    dy = vert_exag*((array[jp1, im1] - array[jm1, im1]) +
                    2.*(array[jp1, i] - array[jm1, i]) +
                    (array[jp1, ip1] - array[jm1, ip1]))/(4.*spacing_y)
    dx = vert_exag*((array[jm1, ip1] - array[jm1, im1]) +
                    2.*(array[j, ip1] - array[j, im1]) +
                    (array[jp1, ip1] - array[jp1, im1]))/(4.*spacing_x)

    return dy, dx


@jit(nopython=True, nogil=True, parallel=True)
def _compute_horn_slope(array, spacing_y, spacing_x):
    '''
    Computes Horn's slope on a raster.
    '''
    slope = np.full(array.shape, np.nan)

    # Interior
    for j in prange(1, array.shape[0] - 1):
        for i in range(1, array.shape[1] - 1):
            dy, dx = _compute_horn_gradient_value(array,
                                                  2*spacing_y, 2*spacing_x,
                                                  j + 1, j, j - 1,
                                                  i + 1, i, i - 1)
            slope[j, i] = math.atan(math.sqrt(dx*dx + dy*dy))
            
    # Borders and corners
    for j in (0, array.shape[0] - 1):
        jp1 = j + 1 if j < array.shape[0] - 1 else array.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, array.shape[1]):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            factor = 2 if i != 0 and i != array.shape[1] - 1 else 1
            dy, dx = _compute_gradient_value(array,
                                             spacing_y, factor*spacing_x,
                                             jp1, j, jm1,
                                             ip1, i, im1)
            slope[j, i] = math.atan(math.sqrt(dx*dx + dy*dy))
    for j in range(1, array.shape[0] - 1):
        for i in (0, array.shape[1] - 1):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            dy, dx = _compute_gradient_value(array,
                                             2*spacing_y, spacing_x,
                                             j + 1, j, j - 1,
                                             ip1, i, im1)
            slope[j, i] = math.atan(math.sqrt(dx*dx + dy*dy))

    return slope


def compute_horn_slope(array, spacing=1):
    '''
    Computes Horn's slope on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.
        
    Returns
    -------
    
    slope : ndarray, shape (n_cells_y, n_cells_x)
        Slope of the raster.
    
    References
    ----------
    
    Horn, B. K. (1981).
    Hill shading and the reflectance map.
    Proceedings of the IEEE, 69(1), 14-47. https://doi.org/10.1109/PROC.1981.11918
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _compute_horn_slope(array, spacing[0], spacing[1])


@jit(nopython=True, nogil=True, parallel=True)
def _compute_horn_aspect(array, spacing_y, spacing_x):
    '''
    Computes Horn's aspect on a raster.
    '''
    aspect = np.zeros(array.shape)

    # Interior
    for j in prange(1, array.shape[0] - 1):
        for i in range(1, array.shape[1] - 1):
            dy, dx = _compute_horn_gradient_value(array,
                                                  2*spacing_y, 2*spacing_x,
                                                  j + 1, j, j - 1,
                                                  i + 1, i, i - 1)
            aspect[j, i] = math.atan2(-dx, dy)
            if aspect[j, i] < 0.:
                aspect[j, i] += 2*math.pi
                
    # Borders and corners
    for j in (0, array.shape[0] - 1):
        jp1 = j + 1 if j < array.shape[0] - 1 else array.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, array.shape[1]):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            factor = 2 if i != 0 and i != array.shape[1] - 1 else 1
            dy, dx = _compute_gradient_value(array,
                                             spacing_y, factor*spacing_x,
                                             jp1, j, jm1,
                                             ip1, i, im1)
            aspect[j, i] = math.atan2(-dx, dy)
            if aspect[j, i] < 0.:
                aspect[j, i] += 2*math.pi
    for j in range(1, array.shape[0] - 1):
        for i in (0, array.shape[1] - 1):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            dy, dx = _compute_gradient_value(array,
                                             2*spacing_y, spacing_x,
                                             j + 1, j, j - 1,
                                             ip1, i, im1)
            aspect[j, i] = math.atan2(-dx, dy)
            if aspect[j, i] < 0.:
                aspect[j, i] += 2*math.pi

    return aspect


def compute_horn_aspect(array, spacing=1):
    '''
    Computes Horn's aspect on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.
        
    Returns
    -------
    
    aspect : ndarray, shape (n_cells_y, n_cells_x)
        Aspect of the raster.
    
    References
    ----------
    
    Horn, B. K. (1981).
    Hill shading and the reflectance map.
    Proceedings of the IEEE, 69(1), 14-47. https://doi.org/10.1109/PROC.1981.11918
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _compute_horn_aspect(array, spacing[0], spacing[1])

################################################################################
# Curvatures 


@jit(nopython=True)
def _compute_plan_curvature_value(array,
                                  spacing_y, spacing_x,
                                  jp1, j, jm1,
                                  ip1, i, im1):
    '''
    Computes the plan curvature value.
    '''
    G = (array[jp1, i] - array[jm1, i])/(2*spacing_y)
    H = (array[j, im1] - array[j, ip1])/(2*spacing_x)
    den = G*G + H*H
    
    if den == 0.:
        return 0.
    
    D = ((array[jm1, i] + array[jp1, i])/2.
         - array[j, i])/(spacing_y*spacing_y)
    E = ((array[j, im1] + array[j, ip1])/2.
         - array[j, i])/(spacing_x*spacing_x)
    F = (-array[jm1, im1]
         + array[jp1, im1]
         + array[jm1, ip1]
         - array[jp1, ip1])/(4*spacing_x*spacing_y)

    return 2*((D*H*H + E*G*G - F*G*H)/den)


@jit(nopython=True, nogil=True, parallel=True)
def _compute_plan_curvature(array, spacing_y, spacing_x):
    '''
    Computes the plan curvature on a raster.
    '''
    curvature = np.zeros(array.shape)

    # Interior
    for j in prange(1, array.shape[0] - 1):
        for i in range(1, array.shape[1] - 1):
            curvature[j, i] = _compute_plan_curvature_value(array,
                                                            spacing_y, spacing_x,
                                                            j + 1, j, j - 1,
                                                            i + 1, i, i - 1)
       
    # Borders and corners
    for j in (0, array.shape[0] - 1):
        jp1 = j + 1 if j < array.shape[0] - 1 else array.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, array.shape[1]):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_plan_curvature_value(array,
                                                            spacing_y, spacing_x,
                                                            jp1, j, jm1,
                                                            ip1, i, im1)
    for j in range(1, array.shape[0] - 1):
        for i in (0, array.shape[1] - 1):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_plan_curvature_value(array,
                                                            spacing_y, spacing_x,
                                                            j + 1, j, j - 1,
                                                            ip1, i, im1)

    return curvature


def compute_plan_curvature(array, spacing=1):
    '''
    Computes the plan curvature on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.
        
    Returns
    -------
    
    curvature : ndarray, shape (n_cells_y, n_cells_x)
        Plan curvature of the raster.
    
    References
    ----------
    
    Zevenbergen, L. W. and Thorne, C. R. (1987).
    Quantitative analysis of land surface topography.
    Earth Surf. Process. Landforms, 12: 47-56. doi:10.1002/esp.3290120107
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _compute_plan_curvature(array, spacing[0], spacing[1])


@jit(nopython=True)
def _compute_coefficients(array, spacing_y, spacing_x, jp1, j, jm1, ip1, i, im1):
    '''
    Computes coefficients for the positive maximal curvature value.
    '''
    a = spacing_y*spacing_x*((array[jm1, im1]
                              + array[jp1, im1]
                              + array[jm1, i]
                              + array[jp1, i]
                              + array[jm1, ip1]
                              + array[jp1, ip1])/6.
                             - (array[j, im1]
                                + array[j, i]
                                + array[j, ip1])/3.)
    b = spacing_y*spacing_x*((array[jm1, im1]
                              + array[j, im1]
                              + array[jp1, im1]
                              + array[jm1, ip1]
                              + array[j, ip1]
                              + array[jp1, ip1])/6.
                             - (array[jm1, i]
                                + array[j, i]
                                + array[jp1, i])/3.)
    c = spacing_y*spacing_x*(array[jp1, im1]
                             + array[jm1, ip1]
                             - array[jm1, im1]
                             - array[jp1, ip1])/4.
    
    return a, b, c


@jit(nopython=True)
def _compute_positive_max_curvature_value(array,
                                          spacing_y, spacing_x,
                                          jp1, j, jm1,
                                          ip1, i, im1):
    '''
    Computes the positive maximal curvature value.
    '''
    a, b, c = _compute_coefficients(array, 
                                    spacing_y, spacing_x, 
                                    jp1, j, jm1, 
                                    ip1, i, im1)
    curvature = -a - b + math.sqrt((a - b)*(a - b) + c*c)
    
    if curvature < 0.:
        return 0.
    return curvature


@jit(nopython=True, nogil=True, parallel=True)
def _compute_positive_max_curvature(array, spacing_y, spacing_x):
    '''
    Computes the positive maximal curvature on a raster.
    '''
    curvature = np.zeros(array.shape)

    # Interior
    for j in prange(1, array.shape[0] - 1):
        for i in range(1, array.shape[1] - 1):
            curvature[j, i] = _compute_positive_max_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    j + 1, j, j - 1,
                                                                    i + 1, i, i - 1)
       
    # Borders and corners
    for j in (0, array.shape[0] - 1):
        jp1 = j + 1 if j < array.shape[0] - 1 else array.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, array.shape[1]):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_positive_max_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    jp1, j, jm1,
                                                                    ip1, i, im1)
    for j in range(1, array.shape[0] - 1):
        for i in (0, array.shape[1] - 1):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_positive_max_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    j + 1, j, j - 1,
                                                                    ip1, i, im1)

    return curvature


def compute_positive_max_curvature(array, spacing=1):
    '''
    Computes the positive maximal curvature on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.
        
    Returns
    -------
    
    curvature : ndarray, shape (n_cells_y, n_cells_x)
        Positive maximal curvature of the raster.
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _compute_positive_max_curvature(array, spacing[0], spacing[1])


@jit(nopython=True)
def _compute_negative_min_curvature_value(array,
                                          spacing_y, spacing_x,
                                          jp1, j, jm1, 
                                          ip1, i, im1):
    '''
    Computes the negative minimal curvature value.
    '''
    a, b, c = _compute_coefficients(array, 
                                    spacing_y, spacing_x, 
                                    jp1, j, jm1, 
                                    ip1, i, im1)
    curvature = -a - b - math.sqrt((a - b)*(a - b) + c*c)
    
    if curvature < 0.:
        return -curvature
    return 0.


@jit(nopython=True, nogil=True, parallel=True)
def _compute_negative_min_curvature(array, spacing_y, spacing_x):
    '''
    Computes the negative minimal curvature on a raster.
    '''
    curvature = np.zeros(array.shape)

    # Interior
    for j in prange(1, array.shape[0] - 1):
        for i in range(1, array.shape[1] - 1):
            curvature[j, i] = _compute_negative_min_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    j + 1, j, j - 1,
                                                                    i + 1, i, i - 1)
       
    # Borders and corners
    for j in (0, array.shape[0] - 1):
        jp1 = j + 1 if j < array.shape[0] - 1 else array.shape[0] - 1
        jm1 = j - 1 if j > 0 else 0
        for i in range(0, array.shape[1]):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_negative_min_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    jp1, j, jm1,
                                                                    ip1, i, im1)
    for j in range(1, array.shape[0] - 1):
        for i in (0, array.shape[1] - 1):
            ip1 = i + 1 if i < array.shape[1] - 1 else array.shape[1] - 1
            im1 = i - 1 if i > 0 else 0
            curvature[j, i] = _compute_negative_min_curvature_value(array,
                                                                    spacing_y, spacing_x,
                                                                    j + 1, j, j - 1,
                                                                    ip1, i, im1)

    return curvature


def compute_negative_min_curvature(array, spacing=1):
    '''
    Computes the negative minimal curvature on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.
        
    Returns
    -------
    
    curvature : ndarray, shape (n_cells_y, n_cells_x)
        Negative minimal curvature of the raster.
    
    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    return _compute_negative_min_curvature(array, spacing[0], spacing[1])

################################################################################
# Wood's multi-scale terrain parameters 


@jit(nopython=True)
def _find_weights(half_size, spacing_y, spacing_x, p):
    '''
    Finds weights for all the cell of a raster based on an inverse distance
    function.
    '''
    window_size = 2*half_size + 1
    weights = np.empty((window_size, window_size))

#     sum_weights = 0.
    for v in range(-half_size, half_size + 1):
        for u in range(-half_size, half_size + 1):
            
            y = spacing_y*abs(v)
            x = spacing_x*abs(u)
            
            distance = np.sqrt(x*x + y*y)
            weights[v + half_size, u + half_size] = 1./(distance + 1)**p
#             sum_weights += weight

    return weights#/sum_weights


@jit(nopython=True)
def _find_normal(half_size, spacing_y, spacing_x, weights):
    '''
    Finds the normal equations for each cell of a raster that allow to fit a 
    quadratic trend surface though a cell's neighbors using least squares.
    '''
    normal = np.empty((6, 6))
    
    x4 = x2y2 = x3y = x3 = x2y = x2 = 0.
    y4 = xy3 = xy2 = y3 = y2 = xy = x1 = y1 = 0.
    N = 0
    for v in range(-half_size, half_size + 1):
        for u in range(-half_size, half_size + 1):
            
            y = v*spacing_y
            x = u*spacing_x
    
            x4 += x*x*x*x*weights[v + half_size, u + half_size]
            x2y2 += x*x*y*y*weights[v + half_size, u + half_size]
            x3y += x*x*x*y*weights[v + half_size, u + half_size]
            x3 += x*x*x*weights[v + half_size, u + half_size]
            x2y += x*x*y*weights[v + half_size, u + half_size]
            x2 += x*x*weights[v + half_size, u + half_size]
            
            y4 += y*y*y*y*weights[v + half_size, u + half_size]
            xy3 += x*y*y*y*weights[v + half_size, u + half_size]
            xy2 += x*y*y*weights[v + half_size, u + half_size]
            y3 += y*y*y*weights[v + half_size, u + half_size]
            y2 += y*y*weights[v + half_size, u + half_size]
            
            xy += x*y*weights[v + half_size, u + half_size]
            
            x1 += x*weights[v + half_size, u + half_size]
            y1 += y*weights[v + half_size, u + half_size]
            
            N += weights[v + half_size, u + half_size]

    normal[0, 0] = x4
    normal[1, 0] = normal[0, 1] = x2y2
    normal[2, 0] = normal[0, 2] = x3y
    normal[3, 0] = normal[0, 3] = x3
    normal[4, 0] = normal[0, 4] = x2y
    normal[5, 0] = normal[0, 5] = x2
    
    normal[1, 1] = y4
    normal[1, 2] = normal[2, 1] = xy3
    normal[1, 3] = normal[3, 1] = xy2
    normal[1, 4] = normal[4, 1] = y3
    normal[1, 5] = normal[5, 1] = y2
    
    normal[2, 2] = x2y2
    normal[2, 3] = normal[3, 2] = x2y
    normal[2, 4] = normal[4, 2] = xy2
    normal[2, 5] = normal[5, 2] = xy
    
    normal[3, 3] = x2
    normal[3, 4] = normal[4, 3] = xy
    normal[3, 5] = normal[5, 3] = x1
    
    normal[4, 4] = y2
    normal[4, 5] = normal[5, 4] = y1
    
    normal[5, 5] = N
    
    return normal


@jit(nopython=True)
def _find_obs(array, j, i, half_size, spacing_y, spacing_x, weights, constrained):
    '''
    Finds the observed vectors for each cell of a raster as part of the normal
    equations for least squares.
    '''
    obs = np.zeros(6)
    
    for v in range(-half_size, half_size + 1):
        for u in range(-half_size, half_size + 1):
            
            y = v*spacing_y
            x = u*spacing_x
            
            obs[0] += weights[v + half_size, u + half_size]*array[j + v, i + u]*x*x
            obs[1] += weights[v + half_size, u + half_size]*array[j + v, i + u]*y*y
            obs[2] += weights[v + half_size, u + half_size]*array[j + v, i + u]*x*y
            obs[3] += weights[v + half_size, u + half_size]*array[j + v, i + u]*x
            obs[4] += weights[v + half_size, u + half_size]*array[j + v, i + u]*y
            if constrained == False:
                obs[5] += weights[v + half_size, u + half_size]*array[j + v, i + u]
                
    return obs


@jit(nopython=True)
def _compute_local_quadratic_model(array,
                                   half_size,
                                   spacing_y,
                                   spacing_x,
                                   p,
                                   constrained):
    '''
    Computes a local quadratic model on each cell of a raster.
    '''
    weights = _find_weights(half_size, spacing_y, spacing_x, p)
    
    normal = _find_normal(half_size, spacing_y, spacing_x, weights)
    normal = np.linalg.pinv(normal)

    a = np.empty((6, array.shape[0] - 2*half_size, array.shape[1] - 2*half_size))
    
    for j in range(half_size, array.shape[0] - half_size):
        for i in range(half_size, array.shape[1] - half_size):
            
            obs = _find_obs(array,
                            j,
                            i,
                            half_size,
                            spacing_y,
                            spacing_x,
                            weights,
                            constrained)
            a[:, j - half_size, i - half_size] = np.dot(normal, obs)
            
    return a


def compute_local_quadratic_model(array,
                                  window_size=3,
                                  spacing=1,
                                  p=0,
                                  constrained=False,
                                  mode='reflect'):
    '''
    Computes a local quadratic model on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.

    window_size : int (default 3)
        Size of the window on which to compute the parameters for each cell of
        the raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    p : float (default 1)
        Exponent defining the distance decay of the weighted least squares to
        determine the best quadratic model fit (e.g., 0: no distance decay;
        1: linear decay; 2: square decay).

    constrained : bool (default False)
        When True, use a constrained quadratic model (i.e., the model becomes an
        exact interpolator at the central cell), otherwise use an unconstrained
        model.

    mode : str (default 'reflect')
        Padding mode. See numpy.pad for the different options.
        
    Returns
    -------
    
    model : ndarray, shape (6, n_cells_y, n_cells_x)
        Coefficients of the quadratic model for each cell of the raster.
    
    References
    ----------
    
    Wood, J. (1996).
    The geomorphological characterisation of digital elevation models.
    Doctoral dissertation, University of Leicester.
    https://leicester.figshare.com/articles/The_geomorphological_characterisation_of_Digital_Elevation_Models_/10152368

    '''
    if isinstance(spacing, (float, int)) == True:
        spacing = (spacing, spacing)

    half_size = int((window_size - 1)/2)

    return _compute_local_quadratic_model(np.pad(array, half_size, mode=mode),
                                          half_size,
                                          *spacing,
                                          p,
                                          constrained)

def _compute_wood_terrain_parameters(model, parameter_type):
    '''
    Computes Wood's multi-scale terrain parameters from a local quadratic model.
    '''
    if parameter_type == 'slope':

        return np.arctan(np.sqrt(model[3]**2 + model[4]**2))
    
    elif parameter_type == 'aspect':
        parameter = np.arctan2(-model[3], model[4])
        parameter[parameter < 0.] += 2*np.pi
        return parameter
    
    elif parameter_type == 'profile curvature':
        parameter = np.zeros((model.shape[1:]))
        g = model[4]**2 + model[3]**2
        parameter[g != 0] = -200*(model[0, g != 0]*model[3, g != 0]**2
                                  + model[1, g != 0]*model[4, g != 0]**2
                                  + model[2, g != 0]*model[3, g != 0]*model[4, g != 0])/(g[g != 0]*(1 + g[g != 0])**1.5)
        return parameter
    
    elif parameter_type == 'plan curvature':
        parameter = np.zeros((model.shape[1:]))
        g = model[4]**2 + model[3]**2
        parameter[g != 0] = 200*(model[1, g != 0]*model[3, g != 0]**2
                                 + model[0, g != 0]*model[4, g != 0]**2
                                 - model[2, g != 0]*model[3, g != 0]*model[4, g != 0])/(g[g != 0]**1.5)
        return parameter
    
    elif parameter_type == 'longitudinal curvature':
        parameter = np.zeros((model.shape[1:]))
        g = model[4]**2 + model[3]**2
        parameter[g != 0] = -2*(model[0, g != 0]*model[3, g != 0]**2
                                + model[1, g != 0]*model[4, g != 0]**2
                                + model[2, g != 0]*model[3, g != 0]*model[4, g != 0])/(g[g != 0])
        return parameter
    
    elif parameter_type == 'cross-sectional curvature':
        parameter = np.zeros((model.shape[1:]))
        g = model[4]**2 + model[3]**2
        parameter[g != 0] = -2*(model[1, g != 0]*model[3, g != 0]**2
                                + model[0, g != 0]*model[4, g != 0]**2
                                - model[2, g != 0]*model[3, g != 0]*model[4, g != 0])/(g[g != 0])
        return parameter

    elif parameter_type == 'maximum curvature':

        return -model[0] - model[1] + np.sqrt((model[0] - model[1])*(model[0] - model[1]) + model[2]*model[2])
    
    elif parameter_type == 'minimum curvature':

        return -model[0] - model[1] - np.sqrt((model[0] - model[1])*(model[0] - model[1]) + model[2]*model[2])
    
    
def compute_wood_terrain_parameters(array,
                                    parameter_type=('slope',
                                                    'aspect',
                                                    'profile curvature',
                                                    'plan curvature',
                                                    'longitudinal curvature',
                                                    'cross-sectional curvature',
                                                    'maximum curvature',
                                                    'minimum curvature'),
                                    window_size=3,
                                    spacing=1,
                                    p=0,
                                    constrained=False,
                                    mode='reflect'):
    '''
    Computes Wood's multi-scale terrain parameters on a raster.
    
    Parameters
    ----------
    
    array : ndarray, shape (n_cells_y, n_cells_x)
        Raster.

    parameter_type : array-like (default ('slope',
                                          'aspect',
                                          'profile curvature',
                                          'plan curvature',
                                          'longitudinal curvature',
                                          'cross-sectional curvature',
                                          'maximum curvature',
                                          'minimum curvature') )
        Parameters to compute. The default value contains all the possible types.

    window_size : int (default 3)
        Size of the window on which to compute the parameters for each cell of
        the raster.
        
    spacing : float or array-like (default 1)
        Spacing between the cells of the raster along each dimension.

    p : float (default 1)
        Exponent defining the distance decay of the weighted least squares to
        determine the best quadratic model fit (e.g., 0: no distance decay;
        1: linear decay; 2: square decay).

    constrained : bool (default False)
        When True, use a constrained quadratic model (i.e., the model becomes an
        exact interpolator at the central cell), otherwise use an unconstrained
        model.

    mode : str (default 'reflect')
        Padding mode. See numpy.pad for the different options.
        
    Returns
    -------
    
    parameters : ndarray, shape (n_parameters, n_cells_y, n_cells_x)
        Parameter values for the raster.
    
    References
    ----------
    
    Wood, J. (1996).
    The geomorphological characterisation of digital elevation models.
    Doctoral dissertation, University of Leicester.
    https://leicester.figshare.com/articles/The_geomorphological_characterisation_of_Digital_Elevation_Models_/10152368

    '''
    model = compute_local_quadratic_model(array,
                                          window_size=window_size,
                                          spacing=spacing,
                                          p=p,
                                          constrained=constrained,
                                          mode=mode)
    
    parameters = np.empty(((len(parameter_type),) + model.shape[1:]))
    for i, ptype in enumerate(parameter_type):
        parameters[i] = _compute_wood_terrain_parameters(model, ptype)
        
    return parameters
