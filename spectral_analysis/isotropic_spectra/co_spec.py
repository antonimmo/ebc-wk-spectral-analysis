def spec_est3(A,B,d1,d2,d3):
    import numpy as np
    l1,l2,l3 = A.shape
    print('before fftn')
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

    Ahat = np.fft.rfftn(window_s*window_t*A)
    Bhat = np.fft.rfftn(window_s*window_t*B)

    cospec = (Bhat*Ahat.conjugate()).real / ((l1*l2*l3)**2) / (df1*df2*df3)
    cospec_dens = np.fft.fftshift(cospec.copy(),axes=(0,1))
    cospec_rms = np.sqrt((cospec_dens[1:,1:,1:-2].sum()*df1*df2*df3))

    return np.fft.fftshift(cospec.real,axes=(0,1)),f1,f2,f3,df1,df2,df3
