def spectral_kinematics(u,v,d1,d2,d3):
    """
    d1,d2 in km
    d2 in hr
    """
    import numpy as np
    l1,l2,l3 = u.shape
    # frequency and wavenumber vectors
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    df3 = 1./(l3*d3)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)
    f3Ny = 1./(2*d3)
    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    f3 = np.arange(0,l3/2+1)*df3

    # spectral window
    # first, the spatial window
    wx = np.matrix(np.hanning(l1))
    wy = np.matrix(np.hanning(l2))
    window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    # now, the time window
    wt = np.hanning(l3)
    window_t = np.repeat(wt,l1*l2).reshape(l3,l2,l1).T

    # wavenumber space
    kx,ky = np.meshgrid(f1/1000,f2/1000)
    kx,ky = kx.T,ky.T
    wvsqr = kx*kx + ky*ky
    print(kx.shape)
    print(ky.shape)
    # Fourier Transform
    uhat = np.fft.rfftn(window_s*window_t*u)
    vhat = np.fft.rfftn(window_s*window_t*v)

    ### Kinetic energy * 2
    modU = uhat*uhat.conjugate()
    modV = vhat*vhat.conjugate()
    KE = (modU + modV)/(df1*df2*df3)/((l1*l2*l3)**2)
    print('uhat shape')
    print(modU.shape)
    j = 0 + 1j

    ### Spectrum of Relative vorticity
    RV = j*kx[:,:,None]*vhat - j*ky[:,:,None]*uhat
    modRV = RV*RV.conjugate()/(wvsqr[:,:,None])/(df1*df2*df3)/((l1*l2*l3)**2)

    ### Spectrum of Divergence
    DV = j*kx[:,:,None]*uhat + j*ky[:,:,None]*vhat
    modDV = DV*DV.conjugate()/(wvsqr[:,:,None])/(df1*df2*df3)/((l1*l2*l3)**2)

    return np.fft.fftshift(KE.real,axes=(0,1)),np.fft.fftshift(modRV.real,axes=(0,1)),np.fft.fftshift(modDV.real,axes=(0,1)),f1,f2,f3
