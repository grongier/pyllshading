"""Colormaps"""

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


from matplotlib import colors


################################################################################
# Yellow-Blue colormap


larkspur_blue = (68/255, 101/255, 137/255)
yucca_yellow = (255/255, 255/255, 190/255)
blue_yellow = colors.LinearSegmentedColormap('blue_yellow',
                                             {'red': [(0., larkspur_blue[0], larkspur_blue[0]),
                                                      (1., yucca_yellow[0], yucca_yellow[0])],
                                              'green': [(0., larkspur_blue[1], larkspur_blue[1]),
                                                        (1., yucca_yellow[1], yucca_yellow[1])],
                                              'blue': [(0., larkspur_blue[2], larkspur_blue[2]),
                                                       (1., yucca_yellow[2], yucca_yellow[2])]})
                                                       