# -*- coding: utf-8 -*-
"""
Created on Feb 9 21:59 2017

@author: htorresg
"""

"""W-F co-spectrum """
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
import co_spec
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
# load 3D array created from matlab script
prnt = '/nobackup/htorresg/reg/wv_ke/'
region = 'KuroshioExt'
season = 'summer'
var1 = 'Theta'
var2 = 'W'
kinema = 'co_spec'
u = h5py.File(prnt+var1+'_'+region+'_summer_z40m.h5','r')
#u = np.load(prnt+var1+'_'+region+'.npz')
u = u['mat'][:,:,:]
#u = np.swapaxes(u,0,2)
#u = np.swapaxes(u,0,1)
v = h5py.File(prnt+var2+'_'+region+'_summer_z40m.h5','r')
#v = np.load(prnt+var2+'_'+region+'.npz')
v = v['mat'][:,:,:]
#v = np.swapaxes(v,0,2)
#v = np.swapaxes(v,0,1)
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
print(iy,ix,it)

# Size of the chunks
# Here you have to specify
# the length of the chunk to be used to compute the 
# spectrum.
# It will depend of what you are interested to see. 
# For instance, if you have hourly outputs, your dt = 1hr.
# If you dataset who has a length of 3 months, it means
# that the length is 30*3*24 = 2160 hours
# Which means that the maximum period resolved by this dataset is
# 2160/2 = 1080 hours (~45 days).
iaux = (45*24) # 45 days x  24 hours = 1080 hours
nt = np.around(it/(iaux),decimals=1) # nt = length(timeseries)/length(chunk)
print(nt)

# Calclate the 3D spectrum
for i in range(int(nt)):
      uaux = u[:,:,i*iaux:i*iaux+iaux]
      vaux = v[:,:,i*iaux:i*iaux+iaux]
      if i == 0:
         Eu,k,l,om,df1,df2,df3 = co_spec.spec_est3(uaux,vaux,2,2,1)
        # Ev,k,l,om = co_spec.spec_est3(vaux,vaux,2,2,1)
      else:
         Eua,_,_,_,_,_,_ = co_spec.spec_est3(uaux,vaux,2,2,1)
        # Eva,_,_,_ = co_spec.spec_est3(vaux,vuax,2,2,1)
         Eu = Eu + Eua
         #Ev = Ev + Eva
Eu = Eu/nt
#print(k.shape)
#------------------------------------
print('------- End WF --------')

#------------------------------------
# isotropic spectrum
I = 0
for i in range(om.size-1):
    kiso,Ei = calc_ispec(k,l,Eu[:,:,i])
    if I == 0:
       Eiso = np.empty((Ei.size,om.size))
    Eiso[:,i] = Ei
    I = I + 1
#=====================================
print('------- End Isotropization --------')

# save
### save the output in python format #####
np.savez(prnt+'/'+kinema+'_wf_'+region+'_'+season+'.npz',
         Eiso=Eiso,kiso=kiso,om=om,df1=df1,df2=df2,df3=df3)
print('------- End save --------')
print('------- Finish --------')
