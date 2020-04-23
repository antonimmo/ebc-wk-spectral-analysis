# -*- coding: utf-8 -*-
"""
Created on October 28 10:22pm

@author: htorresg
"""
"""
  Program to plot the frequency-wavenumber spectrum computed from
  the LLC4320 outputs.
"""
import numpy as np
import sys,glob
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
#::::::::::::::::::::::::::::::::::::::
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['font.family']='Times New Roman'

def coriolis(lat):
    omg = (1.)/24
    return 2*(omg)*np.sin((lat*3.141519)/180)

# :::::::::::: Parent Dir :::::::::
prnt='/Users/hectorg/Documents/JPL_paper/global_wf/files/'
season = 'summer'
var = 'DIV'
dir = prnt+'Spectral_Database/'+season+'/'+var+'/'
name = 'Spectrum_778_at_LON_-59.0_LAT_31.462.nc'
# ::::::::::::::::::

# :::::::::: Open netcdf file ::::::::::
dat = Dataset(dir+name,'r')
kiso = dat['Wvnumber'][:]
omega = dat['Frequency'][:-100]
E = dat['Spectrum'][:,:-100]
print('::::::::: Dimensions :::::::::')
print('Kiso:',kiso.shape)
print('omega: ',omega.shape)
print('E: ',E.shape)

#:::::::::::::::: Plot ::::::::::::::::
if var == 'KE':
    clim = [1e-6,1e-3]
elif var == 'SSH':
    clim = [1e-6,1e-2]
elif var == 'RV':
    clim = [1e-12,1e-10]
elif var == 'DIV':
    clim = [1e-12,1e-10]


##::::::::: Setting the figure :::::::::::
fig = plt.figure(figsize=(7,6))
ax1 = plt.subplot2grid((3,3),(0,1),colspan=2)
ax2 = plt.subplot2grid((3,3),(1,0),rowspan=2)
ax3 = plt.subplot2grid((3,3),(1,1),rowspan=2, colspan=2)


#::::::::: Frequency-Wavenumber spectrum :::::::
cs=plt.pcolor(kiso,omega,
              E.T*kiso[None,...]*omega[...,None],
             cmap='rainbow',norm = LogNorm())
plt.clim(clim)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylim(1/(24*45.),1/3.)
ax3.set_xlim(1/270.,1/5.)
ax3.set_yticks([])
#### Some relevant frequencies
ks = np.array([1.e-3,1.])
f = coriolis(40.411) ## < ========== Latitude from the name
M2 = 1./12.4
K1 = 1./23.93
ax3.plot(ks,[f,f],'k-',linewidth=2.)
ax3.text(1/40.,f+0.002,r'$f$',color='k',size=23)
ax3.plot(ks,[M2,M2],'k--',linewidth=1.)
ax3.text(0.1,M2+.0075,'M2',color='k')
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87,.11,0.01, 0.5])
fig.colorbar(cs, cax=cbar_ax,
             label=r'$K$ $\times$ $\omega$ $\times$ $\Psi(k,\omega)$ [units$^{2}$/(cpkm $\times$ cph)]')


#### :::::: Frequency spectrum :::::::::::
dk = kiso[1]
Ef = E[:,:].sum(axis=0)*dk
ax2.loglog(Ef,omega,linewidth=3.5)
ax2.set_ylim(1/(24.*45.),1/3.)
ax2.invert_xaxis()
ax2.set_ylabel('Frequency [cph]',size=18)


### ::::::: Wavenumber spectrum ::::::::::
domg = omega[1]
Ewv = E[:,:].sum(axis=1)*domg
ax1.loglog(kiso,Ewv,linewidth=3.5)
ax1.set_xlim(1/270.,1/5.)
ax1.set_xticklabels(["",""])
ax1a=ax1.twiny()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_xlabel("Wavelength [km]",size=18)
ax1a.set_xlim(1/270.,1/5.)
ax1a.set_xticks([1./100,1./50,1/10.])
ax1a.set_xticklabels(['100','50','10'])


#:::
plt.show()
