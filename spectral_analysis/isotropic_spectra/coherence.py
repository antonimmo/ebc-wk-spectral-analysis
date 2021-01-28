def cospectrum(A,B,d1,d2,d3):
    import numpy as np
    l1,l2,l3 = A.shape
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    df3 = 1./(l3*d3)
    f1Ny = 1./(2*d1)
    f2Ny = 1./(2*d2)
    f3Ny = 1./(2*d3)
    f1 = np.arange(-f1Ny,f1Ny,df1)
    f2 = np.arange(-f2Ny,f2Ny,df2)
    f3 = np.arange(0,l3/2+1)*df3
    # Spectral window
    # first, the spatial window
    w1 = np.hanning(l1)
    w1 = (l1/(w1**2).sum())*w1
    wx = np.matrix(w1)
    w2 = np.hanning(l2)
    w2 = (l2/(w2**2).sum())*w2
    wy = np.matrix(w2)
    window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    # now, the temporal window
    wt = np.hanning(l3)
    wt = (l3/(wt**2).sum())*wt
    window_t = np.repeat(wt,l1*l2).reshape(l3,l2,l1).T
    # ===== Spectral space ======
    Ahat = np.fft.rfftn(window_s*window_t*A)
    Bhat = np.fft.rfftn(window_s*window_t*B)
    # Cospectrum of A and B
    cs = (Ahat*Bhat.conjugate()).real#/((l1*l2*l3)**2)/(df1*df2*df3)
    #
    # Power spectrum of A
    A = (Ahat*Ahat.conjugate()).real
    # Power spectrum of B
    B = (Bhat*Bhat.conjugate()).real
    #:::::
    ### Coherence #####
    coh = cs/(np.sqrt(A)*np.sqrt(B))
    # zero-padding
    coh = np.fft.fftshift(coh.real,axes=(0,1))
    # ########
    # 
    # cospectrum density
    cs = cs/((l1*l2*l3)**2)/(df1*df2*df3)
    cs = np.fft.fftshift(cs,axes=(0,1))
    #
    # power spectrum density
    A = A/((l1*l2*l3)**2)/(df1*df2*df3)
    A = np.fft.fftshift(A,axes=(0,1))
    B = B/((l1*l2*l3)**2)/(df1*df2*df3)
    B = np.fft.fftshift(B,axes=(0,1))
    return A,B,cs,coh,f1,f2,f3,df1,df2,df3
