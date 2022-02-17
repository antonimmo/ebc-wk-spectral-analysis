import os
import dask
import logging
import numpy as np
import dask.array as da

from uuid import uuid4
from regex_engine import generator

#
from ..common_vars.directories import LUIGI_OUT_FOLDER, POSTPROCESS_OUT_FOLDER
from ..common_vars.regions import face4id
from ..common_vars.time_slices import idx_t
from ..isotropic_spectra.co_spec import cospec_ab,cospec_ab_kh
from ..isotropic_spectra.coherence import coherence_ab
# Spectral analysis
from ..isotropic_spectra.isotropic import calc_ispec

from importlib import reload
from ..luigi_workflows.output import VorticityGrid
#reload(VorticityGrid)


## PATHS
ds_path_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}"
dxdy_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}/{}.txt"
uv_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{0}/{1}/{2}{3:02d}_{4:05d}.npz"
spectra_folder= POSTPROCESS_OUT_FOLDER+"/wk_spectra"
spectra_fn_fmt = "{folder}/{id}{tag}_{t_res}.npz"


##
@dask.delayed(pure=True)
def _var4IdTs(regionId, var_name, t_idx_model, Z_idx, timeRes):
  #print(var_name,t_idx,)
  fname_hf = uv_fname_fmt.format(regionId, timeRes, var_name, Z_idx, t_idx_model)
  try:
    return np.load(fname_hf)["uv"]
  except Exception as err:
    logging.warn("NP load (file: {}) error: {}".format(fname_hf, err))
    return np.load(fname_hf, allow_pickle=True)["uv"]


## Aux function to load vars in parallel using dask
#@dask.delayed
#def _varLoader(xyshape, regionId, var_name, t_slice, Z_idx, timeRes):
#  # Create empty matrix
#  shape = (xyshape[0], xyshape[1], len(t_slice))
#  V = np.zeros(shape)
#  logging.debug("Loading {} shape (k={}): {}".format(var_name, Z_idx, shape))#
#
#  for idx,t in enumerate(t_slice):
#    V_ = _var4IdTs(regionId, var_name, t, Z_idx, timeRes)
#    V[:,:,idx] = V_.compute()
#        
#  return V


class LLCRegion():
  __vars = None
  __grid = None
  __regionId = None
  __timeRes = None
  __tag = None
  __timeVec = None
  t = None
  __face = None
  __nWorkers = 4
  __dt = None
  __dxAvg = None
  __dyAvg = None
  __spectra = None
  
  
  def __init__(self, rid, timeVec, tag=None, t_res="hours", nWorkers=4):
    self.__vars = {}
    self.__regionId = rid
    self.__timeRes = t_res
    self.__timeVec = timeVec
    self.t = timeVec
    self.__tag = "" if tag is None else "_{}".format(tag)
    self.__face = face4id[rid]
    self.__grid = VorticityGrid(rid, t_res)
    self.__nWorkers = nWorkers
    self.__dt = 1 if t_res=="hours" else 24
    self.__dxAvg = np.mean(self.__grid.dxg)/1000
    self.__dyAvg = np.mean(self.__grid.dyg)/1000
    logging.info("Grid: dx = {} km, dy = {} km, dt = {} h".format(self.__dxAvg, self.__dyAvg, self.__dt))
    self.__spectra = dict(self.load_spectra()) if self.spectra_exists() else dict({})
    spec_vars = [k for k in self.__spectra.keys()]
    logging.info("Spectra - Variables: {}".format(spec_vars))

    
  def getGrid(self):
    return self.__grid


  def getGridC(self):
    return self.__grid.xyc()


  def getGridG(self):
    return self.__grid.xyg()


  def varForId(self, var_name, Z_idx=0, t_firstaxis=False):
    shape_uv = self.__grid.f.shape
    shape = (shape_uv[0], shape_uv[1], len(self.__timeVec))
    logging.info("Loading {}: shape (k={}): {}".format(var_name, Z_idx, shape))
  
    loaders = [_var4IdTs(self.__regionId, var_name, t, Z_idx, self.__timeRes) for t in self.__timeVec]
    #sampleLoad = loaders[0].compute()
    arrays = [da.from_delayed(loader, dtype=np.dtype('f'), shape=shape_uv) for loader in loaders]
    print("var4Id type", type(arrays))
    
    r = da.stack(arrays, axis=0) if t_firstaxis else da.stack(arrays, axis=-1)
    print("var4Id type stacked", type(r))

    return r
    

  def loadScalar(self, var_name, Z_idx=0, t_firstaxis=False):
    try:
        self.__vars[var_name]
    except KeyError:
        self.__vars[var_name] = self.varForId(var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)


  def loadHorizontalVector(self, x_var_name, y_var_name, out_var_name, Z_idx=0, t_firstaxis=False):
    try:
        self.__vars[out_var_name]
    except KeyError:
        xVec = self.varForId(x_var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)
        yVec = self.varForId(y_var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)

        # Para face>6, los vectores (U,V) están en las coordenadas "locales"
        # Ver: https://github.com/MITgcm/MITgcm/issues/248 and https://github.com/MITgcm/xmitgcm/issues/204
        if self.__face>6:
           xVec,yVec = yVec,-1*xVec
    
        self.__vars[out_var_name] = (xVec, yVec)


  def load3DVector(self, x_var_name, y_var_name, z_var_name, out_var_name, Z_idx=0, t_firstaxis=False):
    xVec = self.varForId(x_var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)
    yVec = self.varForId(y_var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)
    zVec = self.varForId(z_var_name, Z_idx=Z_idx, t_firstaxis=t_firstaxis)

    # Para face>6, los vectores (U,V) están en las coordenadas "locales"
    # Ver: https://github.com/MITgcm/MITgcm/issues/248 and https://github.com/MITgcm/xmitgcm/issues/204
    if self.__face>6:
       xVec,yVec = yVec,-1*xVec
    
    self.__vars[out_var_name] = (xVec, yVec, zVec)


  def get_uv(self, Z_idx=0, t_firstaxis=False):
    self.loadHorizontalVector("U", "V", "uv", Z_idx, t_firstaxis)


  def get_uvw(self, Z_idx=0, t_firstaxis=False):
    self.load3DVector("U", "V", "W", "uvw", Z_idx, t_firstaxis)


  def get(self, var_name):
    return self.__vars[var_name]


  def set(self, var_, var_name):
    self.__vars[var_name] = var_


  def norm(self, vec_var_name, out_var_name):
    logging.info("Calculating {} = |{}|".format(out_var_name, vec_var_name))
    xVec,yVec = self.get(vec_var_name)
    self.__vars[out_var_name] = np.sqrt(xVec**2 + yVec**2)


  def divergence(self, vec_var_name, out_var_name):
    logging.info("Calculating {} = div({})".format(out_var_name, vec_var_name))
    xVec,yVec = self.get(vec_var_name)
    div = self.__grid.div(np.moveaxis(xVec, -1, 0), np.moveaxis(yVec, -1, 0))
    self.__vars[out_var_name] = np.moveaxis(div, 0, -1)


  def hcurl(self, vec_var_name, out_var_name):
    logging.info("Calculating {} = curl_h({})".format(out_var_name, vec_var_name))
    xVec,yVec = self.get(vec_var_name)
    rot = self.__grid.rv(np.moveaxis(xVec, -1, 0), np.moveaxis(yVec, -1, 0))
    self.__vars[out_var_name] = np.moveaxis(rot, 0, -1)


  def adv_2d(self, out_var_name, uName="U", vName="V", fxName="U", fyName="V", t_firstaxis=False):
    advxName = out_var_name+"_x"
    advyName = out_var_name+"_y"
    if (advxName in self.__vars.keys()) and (advyName in self.__vars.keys()):
      logging.info("({0},{1}) already there!".format(advxName,advyName))
      return self.__vars[advxName], self.__vars[advyName]
    else:
      self.loadScalar(uName)
      self.loadScalar(vName)
      U,V = self.get(uName), self.get(vName)
      logging.info("Calculating {0} = ({1}*grad_x){3} + ({2}*grad_y){3}".format(advxName,uName,vName,fxName))
      advU = self.__grid.adv_2d(U,V,U,t_firstaxis=t_firstaxis)
      logging.info("Calculating {0} = ({1}*grad_x){3} + ({2}*grad_y){3}".format(advyName,uName,vName,fyName))
      advV = self.__grid.adv_2d(U,V,V,t_firstaxis=t_firstaxis)
      self.__vars[advxName] = advU
      self.__vars[advyName] = advV
      return advU, advV


  ##Spectral analysis
  def get_spectrum(self, spectrum_name):
    return self.__spectra[spectrum_name]


  def get_spectra_names(self):
    return self.__spectra.keys()


  def _cospectrum(self, A, B):
    return cospec_ab(A, B, self.__dxAvg, self.__dyAvg, self.__dt)


  def _cospectrum_kh(self, A, B):
    return cospec_ab_kh(A, B, self.__dxAvg, self.__dyAvg)


  def _coh(self, A, B):
    return coherence_ab(A, B, self.__dxAvg, self.__dyAvg, self.__dt)


  def power_spectrum_1d(self, var_name, spectrum_name, recalculate=False):
    if spectrum_name in self.__spectra.keys() and not recalculate:
      logging.info("Spectrum {} already there".format(spectrum_name))
    else:
      logging.info("Calculating {} = FFT_pow({})".format(spectrum_name, var_name))
      hasVar = var_name in self.__vars.keys()
      V = self.__vars[var_name] if hasVar else self.loadScalar(var_name)
      powSpec, kx, ky, omega, dkx, dky, domega = self._cospectrum(V, V)
      kiso, powSpec_iso = calc_ispec(powSpec, kx, ky, omega)
      self.__spectra["k_h"] = kiso
      self.__spectra["om"] = omega
      self.__spectra[spectrum_name] = powSpec_iso
      logging.info("Saved {}({}). min: {}, max: {}".format(spectrum_name, self.__spectra[spectrum_name].shape, np.min(self.__spectra[spectrum_name]), np.max(self.__spectra[spectrum_name])))


  def power_spectrum_2d(self, var_name, spectrum_name, recalculate=False):
    if spectrum_name in self.__spectra.keys() and not recalculate:
      logging.info("Spectrum {} already there".format(spectrum_name))
    else:
      logging.info("Calculating {0} = FFT_pow({1}_x) + FFT_pow({1}_y)".format(spectrum_name, var_name))
      hasVec = var_name in self.__vars.keys()
      if not hasVec:
        logging.error("First load vector into {} -- Vars: {}".format(var_name, self.__vars.keys()))
        return
      Vx, Vy = self.__vars[var_name]
      powSpecX, kx, ky, omega, dkx, dky, domega = self._cospectrum(Vx, Vx)
      powSpecY, kx, ky, omega, dkx, dky, domega = self._cospectrum(Vy, Vy)
      kiso, powSpecX_iso = calc_ispec(powSpecX, kx, ky, omega)
      kiso, powSpecY_iso = calc_ispec(powSpecX, kx, ky, omega)
      self.__spectra["k_h"] = kiso
      self.__spectra["om"] = omega
      self.__spectra["{}_x".format(spectrum_name)] = powSpecX_iso
      self.__spectra["{}_y".format(spectrum_name)] = powSpecY_iso
      self.__spectra[spectrum_name] = powSpecX_iso + powSpecY_iso
      logging.info("Saved {}({}). min: {}, max: {}".format(spectrum_name, self.__spectra[spectrum_name].shape, np.min(self.__spectra[spectrum_name]), np.max(self.__spectra[spectrum_name])))


  def cospectrum(self, var1_name, var2_name, spectrum_name, recalculate=False):
    if spectrum_name in self.__spectra.keys() and not recalculate:
      logging.info("Cospectrum {} already there".format(spectrum_name))
    else:
      logging.info("Calculating {} = FFT({})*FFT({})^*".format(spectrum_name, var1_name, var2_name))
      hasVar1 = var1_name in self.__vars.keys()
      hasVar2 = var2_name in self.__vars.keys()
      if not (hasVar1 and hasVar2):
        logging.error("First load vars ({}, {}) -- Vars: {}".format(var1_name, var2_name, self.__vars.keys()))
        return
      A, B = self.__vars[var1_name], self.__vars[var2_name]
      cospec, kx, ky, omega, dkx, dky, domega = self._cospectrum(A, B)
      kiso, cospec_iso = calc_ispec(cospec, kx, ky, omega)
      self.__spectra["k_h"] = kiso
      self.__spectra["om"] = omega
      self.__spectra[spectrum_name] = cospec_iso
      logging.info("Saved {}({}). min: {}, max: {}".format(spectrum_name, self.__spectra[spectrum_name].shape, np.min(self.__spectra[spectrum_name]), np.max(self.__spectra[spectrum_name])))
      
  def cospectrum_kh(self, var1_name, var2_name, spectrum_name, recalculate=False):
    if spectrum_name in self.__spectra.keys() and not recalculate:
      logging.info("Cospectrum kh {} already there".format(spectrum_name))
    else:
      logging.info("Calculating {}_kh = FFT({})*FFT({})^*".format(spectrum_name, var1_name, var2_name))
      hasVar1 = var1_name in self.__vars.keys()
      hasVar2 = var2_name in self.__vars.keys()
      if not (hasVar1 and hasVar2):
        logging.error("First load vars ({}, {}) -- Vars: {}".format(var1_name, var2_name, self.__vars.keys()))
        return
      A, B = self.__vars[var1_name], self.__vars[var2_name]
      cospec, kx, ky, dkx, dky = self._cospectrum_kh(A, B)
      tVec = np.array(self.__timeVec)
      kiso, cospec_iso = calc_ispec(cospec, kx, ky, tVec)
      self.__spectra["k_h"] = kiso
      self.__spectra["t"] = tVec
      self.__spectra[spectrum_name] = cospec_iso
      logging.info("Saved {}({}). min: {}, max: {}".format(spectrum_name, self.__spectra[spectrum_name].shape, np.min(self.__spectra[spectrum_name]), np.max(self.__spectra[spectrum_name])))


  def coherence(self, var1_name, var2_name, spectrum_name, recalculate=False):
    if spectrum_name in self.__spectra.keys() and not recalculate:
      logging.info("Coherence {} already there".format(spectrum_name))
    else:
      logging.info("Calculating {} = FFT({})*FFT({})^*".format(spectrum_name, var1_name, var2_name))
      hasVar1 = var1_name in self.__vars.keys()
      hasVar2 = var2_name in self.__vars.keys()
      if not (hasVar1 and hasVar2):
        logging.warn("Please first load vars ({}, {}) -- Loaded vars: {}".format(var1_name, var2_name, self.__vars.keys()))
        return
      A, B = self.__vars[var1_name], self.__vars[var2_name]
      logging.info("Calculating coh({},{})".format(var1_name,var2_name))
      coh, kx, ky, omega, dkx, dky, domega = self._coh(A, B)
      kiso, coh_iso = calc_ispec(coh, kx, ky, omega)
      self.__spectra["k_h"] = kiso
      self.__spectra["om"] = omega
      self.__spectra[spectrum_name] = coh_iso
      logging.info("Saved {}({}). min: {}, max: {}".format(spectrum_name, self.__spectra[spectrum_name].shape, np.min(self.__spectra[spectrum_name]), np.max(self.__spectra[spectrum_name])))
      
 
  def coherence_error(self, spectrum_name, nd=1):
    logging.info("Calculating normalized error E[{}]".format(spectrum_name))
    coh_spec = self.__spectra[spectrum_name]
    abs_coh = np.abs(coh_spec)
    sqr_coh = coh_spec**2
    self.__spectra["{}_norm_err".format(spectrum_name)] = np.sqrt(2/nd)*(1-sqr_coh)/abs_coh
    logging.info("Saved E[{}]".format(spectrum_name))
    
  def coh_err_mask(self, spectrum_name, nd=1):
    logging.info("Mask for values where 1>=2*E[{}]".format(spectrum_name))
    norm_err = self.__spectra["{}_norm_err".format(spectrum_name)]
    self.__spectra["{}_err_mask".format(spectrum_name)] = (2*norm_err<=1)
    logging.info("Saved {}_err_mask".format(spectrum_name))


  def spectra_exists(self):
    spectra_fn = spectra_fn_fmt.format(folder=spectra_folder, id=self.__regionId, tag=self.__tag, t_res=self.__timeRes)
    logging.info("Exists? {}: {} -- Filename {}".format(self.__regionId, self.__tag, spectra_fn))
    return os.path.exists(spectra_fn)


  def save_spectra(self):
    spectra_fn = spectra_fn_fmt.format(folder=spectra_folder, id=self.__regionId, tag=self.__tag, t_res=self.__timeRes)
    logging.debug("{}: {} -- Filename {}".format(self.__regionId, self.__tag, spectra_fn))
    self.ensure_dir(spectra_fn)
    np.savez_compressed(spectra_fn, **self.__spectra)
    spec_vars = [k for k in self.__spectra.keys()]
    logging.info("Saved spectra {}: {}".format(spectra_fn, spec_vars))


  def load_spectra(self):
    spectra_fn = spectra_fn_fmt.format(folder=spectra_folder, id=self.__regionId, tag=self.__tag, t_res=self.__timeRes)
    logging.debug("{}: {} -- Opening {}".format(self.__regionId, self.__tag, spectra_fn))
    return np.load(spectra_fn)


## Tools
  def ensure_dir(self, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
      os.makedirs(directory)
