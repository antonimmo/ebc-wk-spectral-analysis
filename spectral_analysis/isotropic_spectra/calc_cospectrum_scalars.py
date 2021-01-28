# -*- coding: utf-8 -*-
"""
Created on Feb 9 21:59 2017

@author: htorresg
"""

"""W-F spectrum hires simulations """
import numpy as np
from numpy import  pi
import os,sys
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
#import co_spec_wind_and_currents as co_spec
#import cross_spec_wind_and_currents as cross_spec
sys.path.append('/nobackup/htorresg/llc_4320/regs/programas/')
import wf_spectrum
import matplotlib.mathtext as mathtext
from matplotlib.colors import LogNorm
from scipy import signal
import h5py
import cospectrum as cospec
import coherence as coh
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
modelo = 'proc/llc4320/' 
prnt = '/nobackup/htorresg/air_sea/'
region = 'Experiment_Kuroshio'
flag = 'surface'
rvn = 'Downwind'
scn = 'SDV'
season = 'summer'
kinema = 'coh_downwind_sdv_'+season
dx = 2.
u = h5py.File(prnt+modelo+rvn+'_'+season+'_'+region+'_'+flag+'.h5','r')
u = u['mat'][2:-2,2:-2,:]
v = h5py.File(prnt+modelo+scn+'_'+season+'_'+region+'_'+flag+'.h5','r')
v = v['mat'][2:-2,2:-2,:]
#taux = h5py.File(prnt+modelo+tauxn+'_'+flag+'_'+region+'.h5','r')
#taux = taux['mat'][1:-1,1:-1,0:-1]
print('Shape')
print(u.shape)

#::::::::::::::::::::::::::::::::::
print('------- End reading --------')

# detrending: space and time
u = signal.detrend(u,axis=0,type='linear')
u = signal.detrend(u,axis=1,type='linear')
u = signal.detrend(u,axis=2,type='linear')
#
v = signal.detrend(v,axis=0,type='linear')
v = signal.detrend(v,axis=1,type='linear')
v = signal.detrend(v,axis=2,type='linear')
#
#taux = signal.detrend(taux,axis=0,type='linear')
#taux = signal.detrend(taux,axis=1,type='linear')
#taux = signal.detrend(taux,axis=2,type='linear')
#
#tauy = signal.detrend(tauy,axis=0,type='linear')
#tauy = signal.detrend(tauy,axis=1,type='linear')
#tauy = signal.detrend(tauy,axis=2,type='linear')
print('------- End detrending --------')


iy,ix,it=u.shape
print(iy,ix,it)
iaux = (30*24)#(60*24*7)/10 #(7*24)
nt = np.around(it/(iaux),decimals=1)
print(nt)

# Calculate the 3D co-spectrum
for i in range(int(nt)):
    uaux = u[:,:,i*iaux:i*iaux+iaux]
    vaux = v[:,:,i*iaux:i*iaux+iaux]
    #txaux = taux[:,:,i*iaux:i*iaux+iaux]
    #tyaux = tauy[:,:,i*iaux:i*iaux+iaux]
    if i == 0:
       Au,Bu,CSu,Eu,k,l,om,df1,df2,df3=coh.cospectrum(uaux,vaux,dx,dx,1)
    else:
       Aua,Bua,CSua,Eua,_,_,_,_,_,_ = coh.cospectrum(uaux,vaux,dx,dx,1)
       Eu = Eu + Eua
       Au = Au + Aua
       Bu = Bu + Bua
       CSu = CSu + CSua
Eu = Eu/nt
Au = Au/nt
Bu = Bu/nt
CSu = CSu/nt
print('------ End co-spectrum ----')

I = 0
for i in range(om.size):
    kiso,Ei = calc_ispec(k,l,Eu[:,:,i])
    kiso,Ai = calc_ispec(k,l,Au[:,:,i])
    kiso,Bi = calc_ispec(k,l,Bu[:,:,i])
    kiso,CSi = calc_ispec(k,l,CSu[:,:,i])
    if I == 0:
       Eiso = np.empty((Ei.size,om.size))
       Aiso = np.empty((Ai.size,om.size))
       Biso = np.empty((Bi.size,om.size))
       CSiso = np.empty((CSi.size,om.size))
    Eiso[:,i] = Ei
    Aiso[:,i] = Ai
    Biso[:,i] = Bi
    CSiso[:,i] = CSi
    I = I + 1
print('-------- End Isotropization --------')

np.savez('/nobackup/htorresg/air_sea/proc/llc4320/'+kinema+'_'+flag+'_'+region+'.npz',
         Coh=Eiso,A=Aiso,B=Biso,COS=CSiso,
         kiso=kiso,om=om,df1=df1,df2=df2,df3=df3)
print('------- End save --------')
print('------- Finish --------')

