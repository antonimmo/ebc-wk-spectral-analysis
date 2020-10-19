def ketrans(u,v,d1,d2,d3):
    import numpy as np
    l1,l2,l3=u.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    df3 = 1./(l3*d3)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)
    f3Ny = 1./(2*d3)
    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f1Ny,df2)
    f3 = np.arange(0,l3/2+1)*df3
    # spectral window
    wx = np.matrix(np.hanning(l1))
    wy = np.matrix(np.hanning(l2))
    window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    # now, the time window
    wt = np.hanning(l3)
    window_t = np.repeat(wt,l1*l2).reshape(l3,l2,l1).T
    ######## End making windows #######
    
    # Wavenumber space
    kx,ky = np.meshgrid(f1,f2)
    wvsqr = kx*kx + ky*ky

    # differentiation in spectral space
    uhat = np.fft.rfftn(u)
    vhat = np.fft.rfftn(v)
    ddx_u = (np.fft.irfftn(i*kx*uhat)).real
    ddy_u = (np.fft.irfftn(i*ky*uhat)).real
    ddx_v = (np.fft.irfftn(i*kx*vhat)).real
    ddy_v = (np.fft.irfftn(i*ky*vhat)).real
    adv_u = u*ddx_u + v*ddy_u
    adv_v = v*ddx_v + v*ddy_v
    Tkxky = (-uhat.conjugate()*np.fft.rfftn(adv_u) - vhat.conjugate()*np.fft.rfftn(adv_v)/(l1*l2*l3)**2).real
    return np.fft.fftshift(Tkxky,axes=(0,1)),f1,f2,f3
