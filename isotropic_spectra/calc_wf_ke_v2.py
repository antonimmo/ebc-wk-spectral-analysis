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

#:::::::::::::::::::::::::::::::::::::::::::
# Gradients of U
# definition: uv vel-comp; d horizontal interval
def gradu(u,v,d):
    uy,ux = np.gradient(u,d,d)
    vy,vx = np.gradient(v,d,d)
    vort = vx - uy
    div = ux + vy
    strain = ((ux-vy)**2 + (vx+uy)**2)**.5
    return vort,div,strain
#:::::::::::::::::::::::::::::::::::::::::::


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
filenameV = 'something' # < =============== change
varname = 'something'   # < =============== change
v = h5py.File(filenameV,'r')
v = v[varname][:,:,:]
v = np.swapaxes(v,0,2)
v = np.swapaxes(v,0,1)
print(u.shape)
#::::::::::::::::::::::::::::::::::
print('------- End reading --------')


# detrend: space and time
u = signal.detrend(u,axis=0,type='linear')
u = signal.detrend(u,axis=1,type='linear')
u = signal.detrend(u,axis=2,type='linear')
v = signal.detrend(v,axis=0,type='linear')
v = signal.detrend(v,axis=1,type='linear')
v = signal.detrend(v,axis=2,type='linear')
print('------- End detrending --------')


iy,ix,it=u.shape
iaux = (20*24)       # chunk
nt = np.around(it/(iaux),decimals=1)


# Calclate the 3D spectrum
for i in range(int(nt)):
      uaux = u[:,:,i*iaux:i*iaux+iaux]
      vaux = v[:,:,i*iaux:i*iaux+iaux]
      if i == 0:
         Eu,k,l,om = wf_spectrum.spec_est3(uaux,dx,dy,dt)
         Ev,k,l,om = wf_spectrum.spec_est3(vaux,dx,dy,dt)
      else:
         Eua,_,_,_ = wf_spectrum.spec_est3(uaux,dx,dx,dt)
         Eva,_,_,_ = wf_spectrum.spec_est3(vaux,dx,dx,dt)
         Eu = Eu + Eua
         Ev = Ev + Eva
Eu = Eu/nt
Ev = Ev/nt
E = 0.5*(Eu + Ev)

print('------- End WF --------')


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
