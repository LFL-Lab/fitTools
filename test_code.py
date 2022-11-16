import numpy as np
import cython 
from numba import jit, njit
from scipy.constants import hbar, e, pi
from scipy.signal import windows, oaconvolve, savgol_filter
from scipy.optimize import curve_fit,leastsq
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import os
from time import perf_counter
from scipy.ndimage.filters import gaussian_filter
import time
import qpFunctionsp

def py_test():
    start = time.time()
    datafile1 = r'G:\Shared drives\LFL\Projects\Quasiparticles\F-drive\Quasiparticles\NBR07_May21_2021_0p47flux_PowerSweep_1p5QP\4.27227GHz_DA30_SR10MHz\NBR07_20210521_165715.bin'
    DATA= qpFunctionsp.loadAlazarData(datafile1)
    bDATA = qpFunctionsp.BoxcarConvolution(DATA,3,10)
    gDATA = qpFunctionsp.GaussianConvolution(DATA,8.4,10)
    h = qpFunctionsp.plotComplexHist(bDATA[0],bDATA[1])
    end = time.time()
    py_time = end-start
    print("Python time = {}".format(py_time))

#def cy_test():
    start = time.time()
    datafile1 = r'G:\Shared drives\LFL\Projects\Quasiparticles\F-drive\Quasiparticles\NBR07_May21_2021_0p47flux_PowerSweep_1p5QP\4.27227GHz_DA30_SR10MHz\NBR07_20210521_165715.bin'
    w.loadAlazarData(datafile1)
    end = time.time()
    cy_time = end-start
    print("Python time = {}".format(cy_time))

py_test()
#cy_test()
