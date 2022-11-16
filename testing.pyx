# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:10:49 2021

@author: lfl
"""

cimport numpy as np
import numpy as np
cimport cython
from scipy.constants import hbar, e, pi
from scipy.signal import windows, oaconvolve, savgol_filter
from scipy.optimize import curve_fit,leastsq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import os
from time import perf_counter
from scipy.ndimage.filters import gaussian_filter

np.import_array()


cdef _gaussian(args,*p,means=None,varis=None):
    '''
    A arbitrary sum of normalized gaussian distributions with scaling amplitudes.

    Parameters
    ----------
    args : tuple (xx,yy)
        xx and yy can either by field points such as np.meshgrid() or IQ data.
    *p : 1-d array-like variable length. MUST BE CALLED WITH PRECEEDING ASTERISK
        asterisk is required so python treats this as multiple arguments rather than a single list.
        a list of parameters for each mode in gaussian distribution.
        for M modes in distribution, should have length M or M*6 per conditions below.
        IF means and varis are None, follows order: amplitude,x0,y0,sigmax,sigmay,angle.
        IF means and varis ARE PROVIDED, then p contains only amplitudes
    means : array with shape (M,2), optional
        the center locations of each mode. each row is [x0,y0] for mode M
    varis : array with shape (M,3), optional
        variances and angle of each mode. each row is [sigma x, sigma y, theta]

    Returns
    -------
    Z : 1-d array
        the values of gaussian distribution at each point.

    '''
    xx,yy = args
    if type(means) is not type(None) and type(varis) is not type(None):
        varis = np.array(varis)
        means = np.array(means)
        if np.ndim(varis) == 1:
            varis = np.array([varis,])
        if np.ndim(means) == 1:
            varis = np.array([means,])
        a = (np.cos(varis[:,2])**2)/(varis[:,0]**2) + (np.sin(varis[:,2])**2)/(varis[:,1]**2)
        b = (-(np.sin(2*varis[:,2]))/(2*varis[:,0]**2) + (np.sin(2*varis[:,2]))/(2*varis[:,1]**2))
        c = (np.sin(varis[:,2])**2)/(varis[:,0]**2) + (np.cos(varis[:,2])**2)/(varis[:,1]**2)
        detInvSIG = a*c-b**2
        z = np.sum([p[i]*np.exp(
            -(a[i]*(means[i,0]-xx)**2 + 
              2*b[i]*(means[i,0]-xx)*(means[i,1]-yy) + 
              c[i]*(means[i,1]-yy)**2)/2
            )/(2*np.pi*np.sqrt(1/detInvSIG[i])) for i in range(0,len(p))],axis=0)
    else:
        z = np.sum([p[i]*np.exp(
            -(((np.cos(p[i+5])**2)/(p[i+3]**2) + (np.sin(p[i+5])**2)/(p[i+4]**2))*(p[i+1]-xx)**2 +
            ((np.sin(p[i+5])**2)/(p[i+3]**2) + (np.cos(p[i+5])**2)/(p[i+4]**2))*(p[i+2]-yy)**2 +
            2*(-(np.sin(2*p[i+5]))/(2*p[i+3]**2) + (np.sin(2*p[i+5]))/(2*p[i+4]**2))*(p[i+1]-xx)*(p[i+2]-yy))/2
            )/(2*np.pi*np.sqrt(1/(
                (((np.cos(p[i+5])**2)/(p[i+3]**2) + (np.sin(p[i+5])**2)/(p[i+4]**2))*
                ((np.sin(p[i+5])**2)/(p[i+3]**2) + (np.cos(p[i+5])**2)/(p[i+4]**2))) - 
                (-(np.sin(2*p[i+5]))/(2*p[i+3]**2) + (np.sin(2*p[i+5]))/(2*p[i+4]**2))**2
                ))
                ) for i in range(0,len(p),6)],axis=0)
        
    return z.ravel('f')