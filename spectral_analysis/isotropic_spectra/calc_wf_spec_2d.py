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
# load 3D array created from matlab script
prnt = '/nobackup/htorresg/reg/'
region = 'kuroshio'
var = 'Etai'
season = 'winter'
data = h5py.File(prnt+var+'_'+region+'_dx2km.mat','r')
#data = np.load(prnt+var+'_'+region+'.npz')
u = data['eta'][:,:,:]
u = np.swapaxes(u,0,2)
u = np.swapaxes(u,0,1)
del data
print(u.shape)
#::::::::::::::::::::::::::::::::::

print('------- End reading ------')

# ----
# esta parte es para cortar la matris
# a una ventana temporal especifica
#de acuerdo a los indices en la
# tercera dimension
# Index:
# winter: 2665:4825
# summer: 7033:9217
u = u[:,:,2665:5945]

u = signal.detrend(u,axis=0,type='linear')
u = signal.detrend(u,axis=1,type='linear')
u = signal.detrend(u,axis=2,type='linear')
print('------ End detrending ---------')

#print('small-matrix')
#print(u.shape)
#-----------------------
# Preambule
iy,ix,it = u.shape
#print(it,iy,ix)
iaux = (20*24)#(60*24*7)/10 #(7*24)
nt = np.around(it/(iaux),decimals=1)
print(nt)

# Calclate the 3D spectrum
for i in range(int(nt)):
      uaux = u[:,:,i*iaux:i*iaux+iaux]
      if i == 0:
         Eu,k,l,om = wf_spectrum.spec_est3(uaux,2,2,1)
      else:
         Eua,_,_,_ = wf_spectrum.spec_est3(uaux,2,2,1)
         Eu = Eu + Eua
Eu = Eu/nt
print('------- End Spectrum --------')
#------------------------------------
# This part is essential, due to the
# isotropization of the spectrum
print(Eu.shape)
print(om.shape)
print(k.shape)
# isotropic spectrum
Eiso = np.empty((126,om.size))

#print(np.shape(k),np.shape(l),np.shape(Eu))

for i in range(om.size-1):
    kiso,Eiso[:,i] = calc_ispec(k,l,Eu[:,:,i])
print('------- End Isospectrum -------')
#=====================================

#::::::::::::::::::::::
# save
np.savez(prnt+'wf/'+var+'_wf_20days_'+region+'_dx2km_'+season+'.npz',
          Eiso=Eiso,kiso=kiso,om=om)
print('------ End save --------')
