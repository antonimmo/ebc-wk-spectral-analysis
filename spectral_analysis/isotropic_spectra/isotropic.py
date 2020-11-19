import numpy as np

import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

#logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.colorbar').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

def centerIdx(length):
    if length%2==1:
        return int((length-1)/2+1)
    else:
        return int(length/2)

def calc_ispec(E,k,l,om):
    """ calculates isotropic spectrum from 3D spectrum """

    logging.debug("Input shapes: E,k,l,f {}".format(E.shape,k.shape,l.shape,om.shape))

    dk,dl = k[1]-k[0],l[1]-l[0]
    l,k = np.meshgrid(l,k)
    wv = np.sqrt(k**2 + l**2)
    logging.debug("Grid shapes: k,l,K {} {} {}".format(k.shape,l.shape,wv.shape))

    if k.max()>l.max():
        kmax = l.max()
        dkr = dl
    else:
        kmax = k.max()
        dkr = dk

    y,x = np.indices(E.shape[:-1])
    logging.debug("Indices shape: x,y {},{}".format(x.shape,y.shape))
    cx = centerIdx(np.max(x)+1)
    cy = centerIdx(np.max(y)+1)
    kr_bins_flat = (np.sqrt((x-cx)**2 + (y-cy)**2)).astype(np.int).ravel() # Medimos la distancia en múltiplos de dkr
    nr = np.bincount(kr_bins_flat)
    nbins = int(np.round(kmax/dkr))
    logging.debug("Nbins {}".format(nbins))

    # create radial wavenumber
    kr =  np.arange(nbins)*dkr+dkr/2 # Bin centers

    for iw in range(om.size):
        E_ = E[:,:,iw]

        # Isotropization w/ radial profile (See https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile)
        Ei = (np.bincount(kr_bins_flat,E_.ravel())/nr)[:nbins]

        # Storage
        if iw == 0:
            Eiso = np.zeros((nbins,om.size))
        Eiso[:,iw] = Ei

    return kr[:nbins], Eiso[:nbins,:]