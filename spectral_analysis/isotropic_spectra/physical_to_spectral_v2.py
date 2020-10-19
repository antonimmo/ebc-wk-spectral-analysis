"""
From physical to spectral space
"""
import numpy as np
import os,warnings,glob
import scipy.io as sci
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import LogNorm
#################################
def cmap_discret(cmap,N):
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0,1.,N),(0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0.,1.,N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key]=[(indices[i],colors_rgba[i-1,ki],colors_rgba[i,ki]) for i in range(N+1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N,cdict,1024)

def win_hann(l1,l2):
    wx=np.matrix(np.hanning(l1))
    wy=np.matrix(np.hanning(l2))
    return np.array(wx.T*wy)

def freq_vec(d1,d2,l1,l2):
    L1=d1*l1
    L2=d2*l2
    df1 = 1./L1
    df2 = 1./L2
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)
    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    k1,k2=np.meshgrid(f1,f2)
    kappa2 = k1**2 + k2**2
    kappa = np.sqrt(kappa2)
    return k1,k2,kappa2,kappa

def fft2(map):
    l1,l2 = map.shape
    wx=np.matrix(np.hanning(l1))
    wy=np.matrix(np.hanning(l2))
    wind_s=np.array(wx.T*wy)
    return (np.fft.fftshift(np.fft.fft2(wind_s*map,axes=(0,1))))

def ifft2(hat):
    return np.real(np.fft.ifft2(np.fft.ifftshift(hat)))
#

#:::::::::::::::::::::::::::::::
prnt = '/Users/hectorg/Documents/JPL_paper/global_wf/docs/Paper_GRL/files/reconstruct/'
name = 'ssh_summer_snap.mat.npz'
data = np.load(prnt+name)
ssh = data['mapa'][:,:]
l1,l2=ssh.shape
#::::::::::::::::::::::::::::::

# :::::: wavenumber grid ::::::::
k1,k2,kappa2,kappa=freq_vec(2,2,l1,l2)
print(k1.shape)
# ::::::::::::::::::::::::::::::

# ::::::: SPectral space :::::::::
sshHat = fft2(ssh)
isshHat = ifft2(sshHat)
print(sshHat.shape)
#::::::::::::::::::::::::::::::::

#::::::::: Filtering based on the transitional scale Lt ::::::::
# Usually, Kappa comes from the transitional scale analysis
# or the scale such that gesotrophic fails in the SSH spectrum.
Kappa = 1./(50)
mask_g = np.zeros(sshHat.shape)
mask_iw = np.zeros(sshHat.shape)
mask_g[kappa<Kappa]=1.
mask_iw[kappa>Kappa]=1.
sshHat_g = sshHat*mask_g
sshHat_iw = sshHat*mask_iw
#:::::::::::::::::::::::::::::::::

#:::::::: Going back to the physical space :::::
ssh_g = ifft2(sshHat_g)
ssh_iw = ifft2(sshHat_iw)
hann = win_hann(l1,l2) # Hanning window ::::::::
ssh_tilde = np.real(ssh_g) + np.real(ssh_iw)
diff = (hann*ssh) - ssh_tilde

# :::::: plot ::::::
cmap=cm.bwr
fig = plt.figure(figsize=(25,12))
plt.subplot(221)
plt.imshow(hann*ssh,vmin=-0.15,vmax=0.15,cmap=cmap_discret(cmap,50),origin='lower')
plt.colorbar(shrink=0.4)
plt.title(r'$\eta_{tot}$',size=30)
plt.subplot(222)
plt.imshow(np.real(ssh_g),vmin=-0.15,vmax=0.15,cmap=cmap_discret(cmap,50),origin='lower')
plt.colorbar(shrink=0.4)
plt.title(r'$\eta_{g}$',size=30)
plt.subplot(223)
plt.imshow(np.real(ssh_iw),vmin=-0.05,vmax=0.05,cmap=cmap_discret(cmap,50),origin='lower')
plt.colorbar(shrink=0.4)
plt.title(r'$\eta_{iw}$',size=30)
plt.subplot(224)
plt.imshow(diff,vmin=-2e-15,vmax=2e-15,cmap=cmap_discret(cmap,50),origin='lower')
plt.colorbar(shrink=0.4)
plt.title(r'$\eta_{tot}$ - $\eta_{est}$',size=30)
plt.show()
