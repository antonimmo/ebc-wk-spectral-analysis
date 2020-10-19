import numpy as np
from numpy import  pi
import os
import warnings
import glob
import scipy.io as sci
import matplotlib
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.mathtext as mathtext
import matplotlib.mathtext as mathtext
from matplotlib.colors import LogNorm
import matplotlib.patheffects as PathEffects
import cmocean

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rc('xtick',labelsize=19)
matplotlib.rc('ytick',labelsize=19)
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['font.family']='Times New Roman'

def coriolis(lat):
    omg = (1.)/24
    return 2*(omg)*np.sin((lat*3.141519)/180)
#
omg_m2 = 1./12.4
omg_f  = 1./24
omg_k1 = 1./23.93
omg_o1 = 1./25.82
ks = np.array([1.e-3,1.])

fig = plt.figure(figsize=(10,11))
clim = [-3e-2,3e-2]

prnt='/Users/hectorg/Documents/JPL_paper/global_wf/files/regional/KE_transfer/'

region = 'KE'
lat = 37

data = np.load(prnt+'KEtrans_'+region+'_winter.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,1)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
plt.ylabel(r'Frequency [cph]',size=18)
#plt.xlabel(r'Horizontal wavenumber [cpkm]',size=18)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-5)
fmax = coriolis(lat+5)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
ax1.text(1/13.,1./1.6,'10 km',size=18)
ax1.text(1/130.,1./1.6,'100 km',size=18)
ax1.text(1/600.,1./1.6,'500 km',size=18)
t = plt.text(1./450,1./(4.),"a) Kuroshio",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100],['',''])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)
plt.text(1./140.,1.5e0,'Jan-Feb-Mar',size=19)


data = np.load(prnt+'KEtrans_'+region+'_summer.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,2)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-8)
fmax = coriolis(lat+8)
#ax1.plot(ks,[f,f],'k-',linewidth=4)
#ax1.text(1/25.,f+0.002,r'$f_{'+str(lat)+'^{o}N}$',color='k',size=23)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
ax1.text(1/13.,1./1.6,'10 km',size=18)
ax1.text(1/130.,1./1.6,'100 km',size=18)
ax1.text(1/600.,1./1.6,'500 km',size=18)
t = plt.text(1./450,1./(4.),"b) Kuroshio",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)
plt.yticks([1e-1,1e-2,1e-3],['','',''])
plt.xticks([1./10,1./100],['',''])
plt.text(1./140.,1.5e0,'Aug-Sept-Oct',size=19)


#==================================
region = 'CC'
lat = 35

data = np.load(prnt+'KEtrans_'+region+'_winter.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,3)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
plt.ylabel(r'Frequency [cph]',size=18)
#plt.xlabel(r'Horizontal wavenumber [cpkm]',size=18)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-8)
fmax = coriolis(lat+8)
#ax1.plot(ks,[f,f],'k-',linewidth=4)
#ax1.text(1/25.,f+0.002,r'$f_{'+str(lat)+'^{o}N}$',color='k',size=23)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
#ax1.text(1/13.,1./1.6,'10 km',size=18)
#ax1.text(1/130.,1./1.6,'100 km',size=18)
#ax1.text(1/900.,1./1.6,'700 km',size=18)
t = plt.text(1./450,1./(4.),"c) NEP",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100],['',''])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)



data = np.load(prnt+'KEtrans_'+region+'_summer.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,4)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-8)
fmax = coriolis(lat+8)
#ax1.plot(ks,[f,f],'k-',linewidth=4)
#ax1.text(1/25.,f+0.002,r'$f_{'+str(lat)+'^{o}N}$',color='k',size=23)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
#ax1.text(1/13.,1./1.6,'10 km',size=18)
#ax1.text(1/130.,1./1.6,'100 km',size=18)
#ax1.text(1/900.,1./1.6,'700 km',size=18)
t = plt.text(1./450,1./(4.),"d) NEP",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)
plt.yticks([1e-1,1e-2,1e-3],['','',''])
plt.xticks([1./10,1./100],['',''])

#====================================
#==================================
region = 'GS'
lat = 30

data = np.load(prnt+'KEtrans_'+region+'_winter.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,5)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
plt.ylabel(r'Frequency [cph]',size=18)
plt.xlabel(r'Horizontal wavenumber [cpkm]',size=18)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-8)
fmax = coriolis(lat+8)
#ax1.plot(ks,[f,f],'k-',linewidth=4)
#ax1.text(1/25.,f+0.002,r'$f_{'+str(lat)+'^{o}N}$',color='k',size=23)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
#ax1.text(1/13.,1./1.6,'10 km',size=18)
#ax1.text(1/130.,1./1.6,'100 km',size=18)
#ax1.text(1/900.,1./1.6,'500 km',size=18)
t = plt.text(1./450,1./(4.),"e) AC",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)

data = np.load(prnt+'KEtrans_'+region+'_summer.npz')
kiso = data['kiso'][1:]
Eiso = data['Eiso'][1:,1:]
om = data['om'][1:]
E = Eiso*kiso[:,None]*om[None,:]
ax1 = fig.add_subplot(3,2,6)
cs=plt.pcolormesh(kiso,om,E.T,shading='flat',
             cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.clim(clim)
plt.xlabel(r'Horizontal wavenumber [cpkm]',size=18)
ax1.set_ylim(1/(24*45.),1/2.)
ax1.set_xlim(1/450.,1/4.)
f = coriolis(lat)
fmin = coriolis(lat-8)
fmax = coriolis(lat+8)
#ax1.plot(ks,[f,f],'k-',linewidth=4)
#ax1.text(1/25.,f+0.002,r'$f_{'+str(lat)+'^{o}N}$',color='k',size=23)
ax1.plot(ks,[omg_m2,omg_m2],'k--',linewidth=1.)
ax1.text(0.1,omg_m2+.0075,'M2',color='k')
ax1.plot(ks,[omg_k1,omg_k1],'k--',linewidth=1.)
ax1.plot(ks,[omg_o1,omg_o1],'k--',linewidth=1.)
ax1.text(0.1,omg_k1+.002,'K1',color='k')
ax1.text(0.1,omg_o1-.0075,'O1',color='k')
#ax1.text(1/13.,1./1.6,'10 km',size=18)
#ax1.text(1/130.,1./1.6,'100 km',size=18)
#ax1.text(1/900.,1./1.6,'700 km',size=18)
t = plt.text(1./450,1./(4.),"f) AC",fontsize=18,
             path_effects=[PathEffects.withStroke(linewidth=2,
             foreground="w")])
#t = plt.text(1./550.,1./4.,'JFM',fontsize=22,
#             path_effects=[PathEffects.withStroke(linewidth=2,
#             foreground="w")])
plt.xticks([1./10,1./100])
plt.fill_between([1./700,1./2],fmin,fmax,color='k',alpha=.25)
plt.yticks([1e-1,1e-2,1e-3],['','',''])
plt.xticks([1./10,1./100])

cbar_ax = fig.add_axes([0.9,0.3, 0.015, 0.45])
fig.colorbar(cs, ticks=[-3e-2,0,3e-2],cax=cbar_ax,
          label=r'$K$ $\times$ $\omega$ $\times$ $(-Re[\widehat{\mathbf{u}^{*}_{h}}\cdot(\widehat{adv})])/[cph \times cpkm]$')
cbar_ax.set_yticklabels(['-3e-2', '0', '3e-2'])
#========

#----
prnt_out = '/Users/hectorg/Documents/JPL_paper/global_wf/docs/Paper_GRL/figures/'
plt.savefig(prnt_out+'KETRANS_WF_spectrum3.png',
            format='png',dpi=550,facecolor='w',
            bbox_inches='tight')
#=========================

plt.show()
