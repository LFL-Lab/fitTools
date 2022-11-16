# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:29:33 2020

change log:
    4/13/2021 - added functions to import data in smaller chunks. 
        changed loadAlazarData to return data in uint16 format rather than convert to mV.
        added uint16_to_mV to convert uint16 data to mV depending on whether 12 bit 
            sample is stored in least or most significant digits


@author: lfl
"""
import cython
import numpy as np
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

cpdef expectedShift(q0, f0, ls, phi, delta=*)
cpdef f_n_phi(phi, n, L=*, C=*, Ls=*,Delta=*)
cpdef get_Ls(q0, L)

cpdef uint16_to_mV(data, bitshifted=*)
   
cpdef getNumberSegmentsForBigData(fpath, segmentSizeGB=*)

cpdef loadChunk(fpath, elementsPerSegment, segment, nChannels=*)

cpdef loadAlazarData(fpath, nChannels=*)

# cpdef loadAlazarDataCppAvg(fpath,nChannels=*):
#     '''pulls ATS9371 digitizer data from .bin save file (fpath) and returns array in mV units.
    
#     returns DATA array in mV.
#     if nChannels = 2, returns (#Samples,2) array. Otherwise 1 dimension array
#     ------------------------
#     fpath:      file path to .bin file
#     nChannels:  number of channels which fpath file includes
#     '''
#     if fpath[-4:] != '.bin':
#         print('fpath should be full path to .bin filetype of raw digitizer data.')
#         return None
#     int DATA = (np.fromfile(fpath,dtype=np.uint16)- 32767.5) * (800/65536)
#     if nChannels == 2:
#         DATA = DATA.reshape((2,len(DATA)//2))
#     return DATA

cpdef HannConvolution(data,avgTime,sampleRate)
cpdef FlattopConvolution(data,avgTime,sampleRate)
cpdef GaussianConvolution(data,avgTime,sampleRate)

cpdef BoxcarConvolution(data,avgTime,sampleRate)
   
cpdef BoxcarDownsample(data,avgTime,sampleRate,returnRate=*)

# cdef BoxcarDownsampleCppAvg(data,avgTime,sampleRate,returnRate=False):
#     '''Integrates data by boxcar averaging blocks of duration avgTime.
    
#     returns integrated data with reduced shape and optionally the reduced sample rate.
#     -------------------------------------
#     data:       single or dual channel data. dual channel should have shape (nSamples,2)
#     avgTime:    duration of boxcar window in seconds
#     sampleRate: sample rate of data in Hz.
#     returnRate: boolean, cpdefault False. If True, returns the new rate, which may 
#                     be slightly off from 1/avgTime due to rounding.
#     '''
#     nAvg = int(max(avgTime*sampleRate,1))
#     if len(data.shape) == 2:
#         nSamples = data.shape[1]
#         data2 = data[:,:(nSamples//nAvg)*nAvg].reshape((2,nSamples//nAvg,nAvg))
#     else:
#         nSamples = len(data)
#         data2 = data[:(nSamples//nAvg)*nAvg].reshape((nSamples//nAvg,nAvg))
#     if returnRate:
#         return np.mean(data2,axis=-1), sampleRate/nAvg
#     return np.mean(data2,axis=-1)

cpdef plotComplexHist(I,Q,bins=*,figsize=*,returnHistData=*)
cpdef make_ellipses(gmm,ax,colors)
cpdef make_ellipses2(means,varis,ax,colors)
cpdef getGaussianSNR(gmm,mode1=*,mode2=*)

cpdef getDataSNR(mode1data,mode2data)

cpdef predictWithBayes(gmm,data,nMemory=*)
cpdef extractTimeBetweenTrapEvents(nEst,time)
cpdef extractLifetimes(nEst,time)
cpdef extractAntiLifetimes(nEst,time)

cpdef extractDurations(nEst,time)

cpdef plotTauDist(dist,bins=*,color=*,alpha=*,figsize=*)
cpdef plotTimeSeries(data,nEst,time,start,stop,zeroTime=*)
cpdef extractLifetimesWithModes(nEst,time)
cpdef extractLifetimesWithTwoModes(nEst,time)
cpdef exp(t,a,tau)
cpdef fitExpDecay(dist,t,cut=*,returnTauDetector=*,returnSGDIST=*)
  
cpdef fitAndPlotExpDecay(dist,cut=*,returnTaus=*,figsize=*,bins=*)

cpdef plotBurstSearch(nEst,avgTime,sampleRate,method=*)

cpdef getBurstIndices(burstSearch,threshold,nConsecutive=*)
cpdef getRates(nEst,sampleRate)

cpdef getRollingRates(nEst,sampleRate,windowDuration)

# cdef inline gaussian(widths,means):
#     """Returns a gaussian function with the given parameters"""
#     width_x = float(widths[0])
#     width_y = float(widths[1])
#     return lambda x,y: np.exp(
#                 -(((means[0]-x)/width_x)**2+((means[1]-y)/width_y)**2)/2
#                 )/(2*np.pi*width_x*width_y)

cpdef getGaussian(args,p,means=*,varis=*)

cpdef fitGaussian(histDATA,guess,means=*,varis=*)
  
cpdef fitGaussiansIndependent(quietSubs)
   
cpdef getGaussianProbabilities(DATA,amps,means,varis)
cpdef predictWithBayes2(probs,nMemory=*)

cpdef _gaussian(args,p,means=*,varis=*)
cpdef getQuietSubsets(DATA,nEst,time,thresholds)

cpdef getQuietSubsets2(DATA,nEst,nSubs=*)

cpdef plotQuietSubsets(quietSubs)
cpdef getQuietThresholds(lifetimes)

# cpdef gaussianMix(heights,widths,means):
#     return lambda x,y: np.sum([h*gaussian(w,m)(x,y)for h,m,w in zip(heights,means,widths)],axis=0)

# cpdef fitgaussian(histDATA,widths,means):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution found by a fit"""
#     xx,yy = np.meshgrid((histDATA[1][:-1]+histDATA[1][1:])/2,(
#         histDATA[2][:-1]+histDATA[2][1:])/2)
#     params = [widths,means]
#     p0 = np.repeat(np.max(histDATA[0]),len(means))
#     callGaussian = lambda d,*p: np.ravel(gaussianMix(np.array(p),*params)(xx,yy))
#     # errorfunction = lambda p,params: np.ravel(gaussianMix(
#     #     p,*params)(xx,yy) - histDATA[0])
#     # p, success = leastsq(errorfunction, p0, args = params)
#     pars,cov = curve_fit(callGaussian,np.array([xx,yy]),histDATA[0].ravel(),p0=p0)
#     return xx,yy,pars

cpdef subtractionGaussianFit(DATA,nModes)
cpdef _fitGaussianMasked(mask,histDATA,guess,means=*,varis=*)
cpdef PoissonCorrection(ti,td,tinot,tdnot)