# -*- coding: utf-8 -*-
"""
Created on Feb 9 21:59 2017

@author: htorresg
"""

"""W-F spectrum hires simulations """
import numpy as np
from numpy import  pi
import os
import warnings
import glob
import scipy.io as sci
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.mathtext as mathtext
import wf_spectrum
import matplotlib.mathtext as mathtext
from matplotlib.colors import LogNorm
from scipy import signal
import h5py
#========================================

#:::::::::::::::::::::::::::::::::::::::::::::::
def calc_ispec(k,l,E):
    """ calculates isotropic spectrum from 2D spectrum """

    dk,dl = k[1,]-k[0],l[1]-l[0]
    l,k = np.meshgrid(l,k)
    wv = np.sqrt(k**2 + l**2)

    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()

    # create radial wavenumber
    dkr = np.sqrt(dk**2 + dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr,dkr)
    ispec = np.zeros(kr.size)
    #print(ispec.shape)
    #print(kr.shape)
    for i in range(kr.size):
        fkr =  (wv>=kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
    #    print(fkr.shape)
        dth = pi / (fkr.sum()-1)
        ispec[i] = E[fkr].sum() * kr[i] * dth

    return kr, ispec
#:::::::::::::::::::::::::::::::::::::::::



#:::::::::::::::::::::::::::::::::::::::::

"""
Frequency-Wavenumber spectrum for a scalar quantuty

$\hat{\phi}(k,l,\omega)$ = $F[phi(x,y,t)]$
"""

#:::::::::::::::::::::::::::::::::::::::::
## Parameters 
dx = N   # < =============== change
dy = M   # < =============== change
dt = L   # < =============== change
output_directory = 'something' # < =============== change
output_name = 'something'      # < =============== change


# load 3D array created from matlab script
filenameU = 'something'  # < =============== change
varname = 'something'    # < =============== change
u = h5py.File(filenameU,'r')
u = u[varname][:,:,:]
u = np.swapaxes(u,0,2)
u = np.swapaxes(u,0,1)
#::::::::::::::::::::::::::::::::::
print('------- End reading --------')


# detrend: space and time
u = signal.detrend(u,axis=0,type='linear')
u = signal.detrend(u,axis=1,type='linear')
u = signal.detrend(u,axis=2,type='linear')
print('------- End detrending --------')


iy,ix,it=u.shape
iaux = (20*24)       # chunk
nt = np.around(it/(iaux),decimals=1)

# Calclate the 3D spectrum
for i in range(int(nt)):
      uaux = u[:,:,i*iaux:i*iaux+iaux]
      if i == 0:
         Eu,k,l,om = wf_spectrum.spec_est3(uaux,dx,dy,dt)
      else:
         Eua,_,_,_ = wf_spectrum.spec_est3(uaux,dx,dy,dt)
         Eu = Eu + Eua
Eu = Eu/nt
print('------- End Spectrum --------')


#------------------------------------
# isotropic spectrum
I = 0
for i in range(om.size-1):
    kiso,Ei = calc_ispec(k,l,Eu[:,:,i])
    if I == 0:
        Eiso = np.empty((len(Ei),om.size))
    Eiso[:,i] = Ei
#=====================================
print('------- End Isotropization --------')


# save
np.savez(output_directory+output_name+'.npz',
         Eiso=Eiso,kiso=kiso,om=om)
print('------- End save --------')
print('------- Finish --------')
