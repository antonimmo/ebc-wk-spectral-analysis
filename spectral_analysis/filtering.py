import logging
import numpy as np


def kxky(dx,dy,shape):
    Ny,Nx = shape[-2:] # Se toman las últimas 2
    logging.info("kxky -- Nx: {}, Ny: {}".format(Nx,Ny))
    Lx = Nx*dx
    Ly = Ny*dy
    logging.info("kxky -- Lx: {}, Ly: {}".format(Lx,Ly))
    delta_kx = 1/Lx
    delta_ky = 1/Ly
    N_kx = (Nx-1)/2
    N_ky = (Ny-1)/2
    Kx = delta_kx*np.arange(-N_kx,N_kx+1)
    Ky = delta_ky*np.arange(-N_ky,N_ky+1)
    return np.meshgrid(Kx,Ky)


def create_filter_k(Lt_km=50,order=0,k_=None,flip=False): #Orden 0 es el filtro ideal, de 1 de adelante es el Butterworth
    K_max = 1/Lt_km
    kx_,ky_ = k_

    if order==0:
        filter_mask_plt = (np.square(kx_)+np.square(ky_) <= K_max**2).astype(np.float32)
    else:
        with np.errstate(over='ignore'):
            filter_mask_plt = 1/( 1 + ( (np.square(kx_)+np.square(ky_))/(K_max**2) )**order )

    filter_mask = np.fft.fftshift(filter_mask_plt)

    if flip:
        filter_mask = filter_mask*np.fliplr(filter_mask) # Simetria en Kx
        filter_mask = filter_mask*np.flipud(filter_mask) # Simetria en Ky

    #plt.figure()
    #plt.pcolormesh(kx_,ky_,np.fft.fftshift(filter_mask))
    #plt.colorbar()
    #plt.xscale('symlog',linthreshx=1/500,linthreshy=1/500)
    #plt.yscale('symlog',linthreshx=1/500,linthreshy=1/500)
    #plt.show()
    return filter_mask


# Orden 0 es el filtro ideal, de 1 de adelante es el Butterworth
def create_filter_om(Nt, cuttoff_h=30, dt_h=1.0, order=0, flip=False):
    # 
    Lt = Nt*dt_h
    delta_om = 1/Lt
    N_om = (Nt-1)/2
    om = delta_om*np.arange(-N_om,N_om+1)
    #
    Om_max = 1/cuttoff_h
    
    if order==0:
        filter_mask_plt = (np.square(om) <= Om_max**2).astype(np.float32)
    else:
        with np.errstate(over='ignore'):
            filter_mask_plt = 1/( 1 + ( np.square(om)/(Om_max**2) )**order )
    
    filter_mask = np.fft.fftshift(filter_mask_plt)

    if flip:
        filter_mask = filter_mask*np.flip(filter_mask) # Simetria en Omega

    #plt.figure()
    #plt.plot(kx_,ky_,np.fft.fftshift(filter_mask))
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    #plt.show()
    return filter_mask


def filter_fft_butterworth_kh(var_xy,Lt_km=50,dx=2,dy=2,order=100):
    var_k = np.fft.fft2(var_xy) # fft2 corre sobre los dos últimos ejes (-2,-1)
    
    k_ = kxky(dx,dy,var_xy.shape)
    filter_mask = create_filter_k(Lt_km=Lt_km,order=order,k_=k_,flip=True)
    
    # El cálculo de la inversa se hará sobre la variable a la que no se le aplica fftshift,
    # ya que np.flipup y np.fliplr no conmutan con fftshift
    var_k_lo = var_k*filter_mask
    _var_lo = np.fft.ifft2(var_k_lo)
    
    # Eliminamos la parte imaginaria, ya que es espuria
    var_lo = np.real(_var_lo)
    var_hi = var_xy-var_lo
    
    return var_lo,var_hi


def filter_fft_butterworth_omega(var_txy,cuttoff_h=30,dt_h=1,order=100,axis=-1):
    logging.info("omega BW -- shape: {}".format(var_txy.shape))
    var_om = np.fft.fft(var_txy, axis=axis) # fft en el tiempo (primer axis)
    
    filter_mask = create_filter_om(var_txy.shape[axis], cuttoff_h=cuttoff_h, dt_h=dt_h, order=order, flip=True)
    
    # El cálculo de la inversa se hará sobre la variable a la que no se le aplica fftshift,
    # ya que np.flipup y np.fliplr no conmutan con fftshift
    var_om_lo = var_om*filter_mask
    _var_lo = np.fft.ifft(var_om_lo,axis=axis)
    
    # Eliminamos la parte imaginaria, ya que es espuria
    var_lo = np.real(_var_lo)
    var_hi = var_txy-var_lo
    
    return var_lo,var_hi
