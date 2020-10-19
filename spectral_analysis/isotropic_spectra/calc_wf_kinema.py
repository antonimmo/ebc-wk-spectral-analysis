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
# load 3D array created from matlab script
prnt = '/nobackup/htorresg/reg/'
region = 'CalCoast'
season = 'winter'
var1 = 'Ui'
var2 = 'Vi'
variable = 'vorticity'
u = h5py.File(prnt+var1+'_'+region+'_dx2km.mat','r')
u = u['eta'][:,:,:]
u = np.swapaxes(u,0,2)
u = np.swapaxes(u,0,1)
v = h5py.File(prnt+var2+'_'+region+'_dx2km.mat','r')
v = v['eta'][:,:,:]
v = np.swapaxes(v,0,2)
v = np.swapaxes(v,0,1)
#u = np.load(prnt+var1+'_'+region+'.npz')
#u = u['mat'][:,:,:]
#v = np.load(prnt+var2+'_'+region+'.npz')
#v = v['mat'][:,:,:]
print(u.shape)
#::::::::::::::::::::::::::::::::::
print('------- End reading --------')

#-------
# esta parte es para cortar la serie de tiempo
# index:
# winter: 2665:4825
# summer: 7033:9217
if season == 'summer':
   u = u[:,:,7033:9217]
   v = v[:,:,7033:9217]
else:
   u = u[:,:,2665:4825]
   v = v[:,:,2665:4825]
# ---

# detrend: space and time
u = signal.detrend(u,axis=0,type='linear')
u = signal.detrend(u,axis=1,type='linear')
u = signal.detrend(u,axis=2,type='linear')
v = signal.detrend(v,axis=0,type='linear')
v = signal.detrend(v,axis=1,type='linear')
v = signal.detrend(v,axis=2,type='linear')
print('------- End detrending --------')

# ------------- Kinema prop --------
iy,ix,it = u.shape
print(iy,ix,it)
var = np.empty([iy,ix,it])
#print(strai.shape)
for ii in range(0,it):
  #print(ii)
  if variable == 'strain':
     xi,div,var[:,:,ii] = gradu(u[:,:,ii],v[:,:,ii],2*1000)
  elif variable == 'divergence':
     xi,var[:,:,ii],strai = gradu(u[:,:,ii],v[:,:,ii],2*1000)
  elif variable == 'vorticity':
     var[:,:,ii],div,strai = gradu(u[:,:,ii],v[:,:,ii],2*1000)
#-------------------------------------
print('------- End kinema compute --------')

#print(it,iy,ix)
iaux = (45*24)#(60*24*7)/10 #(7*24)
nt = np.around(it/(iaux),decimals=1)
print(nt)

# Calclate the 3D spectrum
for i in range(int(nt)):
    uaux = var[:,:,i*iaux:i*iaux+iaux]
    if i == 0:
       Eu,k,l,om = wf_spectrum.spec_est3(uaux,2,2,1)
    else:
       Eua,_,_,_ = wf_spectrum.spec_est3(uaux,2,2,1)
       Eu = Eu + Eua
Eu = Eu/nt
#------------------------------------
print('------- End WF --------')

#------------------------------------
# isotropic spectrum
Eiso = np.empty((113,om.size))
# save the 2D spectrum
#np.savez(fname,k=k,l=l,Eu=Eu)
for i in range(om.size-1):
    kiso,Eiso[:,i] = calc_ispec(k,l,Eu[:,:,i])
#=====================================
print('------- End Isotropization --------')

# save
np.savez(prnt+'wf/'+variable+'_wf_'+region+'_'+season+'.npz',
         Eiso=Eiso,kiso=kiso,om=om)
print('------- End save --------')
print('------- Finish --------')
