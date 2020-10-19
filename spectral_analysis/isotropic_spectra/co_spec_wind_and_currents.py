def cospectrum(taux,tauy,u,v,d1,d2,d3):
    import numpy as np
    l1,l2,l3 = u.shape
    print(l1,l2,l3)
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

    tauxhat = np.fft.rfftn(window_s*window_t*taux)
    tauyhat = np.fft.rfftn(window_s*window_t*tauy)
    uhat = np.fft.rfftn(window_s*window_t*u)
    vhat = np.fft.rfftn(window_s*window_t*v)

    ### Co-spectrum -----
    taux_u = (tauxhat*uhat.conjugate()).real/((l1*l2*l3)**2)/(df1*df2*df3)
    tauy_v = (tauyhat*vhat.conjugate()).real/((l1*l2*l3)**2)/(df1*df2*df3)
    cospec =  tauy_v 
    return np.fft.fftshift(cospec.real,axes=(0,1)),f1,f2,f3,df1,df2,df3
