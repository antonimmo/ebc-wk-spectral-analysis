import numpy as np

def spec_est3(A,d1,d2,d3,varname,beta=3.86):
    l1,l2,l3 = A.shape
    N = A.size
    df1 = 1./(l1*d1)
    df2 = 1./(l2*d2)
    df3 = 1./(l3*d3)
    #f1Ny = 1./(2*d1)
    #f2Ny = 1./(2*d2)
    #f3Ny = 1./(2*d3)
    #f1 = np.arange(-f1Ny,f1Ny,df1)
    #f2 = np.arange(-f2Ny,f2Ny,df2)
    #f3 = np.arange(0,l3/2+1)*df3
    f1 = np.fft.fftshift(np.fft.fftfreq(l1,d1))
    f2 = np.fft.fftshift(np.fft.fftfreq(l2,d2))
    f3 = np.fft.rfftfreq(l3,d3)

    # spectral window -- Kaiser window
    # https://dsp.stackexchange.com/questions/40598/why-would-one-use-a-hann-or-bartlett-window
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.kaiser.html
    # first, the spatial window
    #beta = 4.86
    wx = np.matrix(np.kaiser(l1,beta))
    wy = np.matrix(np.kaiser(l2,beta))
    window_s = np.repeat(np.array(wx.T*wy),l3).reshape(l1,l2,l3)
    # now, the time window
    wt = np.kaiser(l3,beta)
    window_t = np.repeat(wt,l1*l2).reshape(l3,l2,l1).T
    Ahat = np.fft.rfftn(window_s*window_t*A)
    Aabs = 2 * (np.abs(Ahat)**2) #/ (df1*df2*df3) / ((l1*l2*l3)**2)

    ## Parseval's
    # Energy in original domain
    e = np.sum(A**2)#*d1*d2*d3
    print(varname+" Energy (x,y,t)",e)
    #Energy in FFT domain
    spec_e = np.sum(Aabs)/N #*df1*df2*df3
    print(varname+" Energy (k,l,f)",spec_e)
    # Energy ratio
    ratio_e = e/spec_e
    # Correting
    #Aabs = Aabs*ratio_e
    #print(varname+" Corrected energy (k,l,f)",np.sum(Aabs)/N)

    ## PSD
    print("Dividing by N (total # of elems) to get (approximate) PSD")
    Aabs = Aabs/N

    return np.fft.fftshift(Aabs,axes=(0,1)),f1,f2,f3
