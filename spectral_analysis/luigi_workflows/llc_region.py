import logging
import numpy as np
from multiprocessing import Pool
#
from ..common_vars.directories import LUIGI_OUT_FOLDER
from ..common_vars.time_slices import idx_t
from ..common_vars.regions import face4id
from .output import VorticityGrid

## PATHS
ds_path_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}"
dxdy_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}/{}.txt"
uv_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{0}/{1}/{2}{3:02d}_{4:05d}.npz"

class LLCRegion():
  __vars = None
  __grid = None
  __regionId = None
  __timeRes = None
  __timeVec = None
  __face = None


  def __init__(self, rid, timeVec, t_res="hours"):
    self.__vars = {}
    self.__regionId = rid
    self.__timeRes = t_res
    self.__timeVec = timeVec
    self.__face = face4id[rid]
    self.__grid = VorticityGrid(rid, t_res)


  def getGridC(self):
    return self.__grid.xyc


  def getGridG(self):
    return self.__grid.xyg


  def _var4IdTs(self, var_name, t_idx, Z_idx):
    print(var_name,t_idx,)
    fname_hf = uv_fname_fmt.format(self.__regionId, self.__timeRes, var_name, Z_idx, t_idx)
    v = np.load(fname_hf)["uv"]

    return v


  def varForId(self, var_name, Z_idx=0, t_firstaxis=False):
    # Create empty matrix
    shape_uv = self.__grid.f.shape
    shape = (shape_uv[0], shape_uv[1], len(self.__timeVec))
    logging.info("Loading {}: shape (k={}): {}".format(var_name, Z_idx, shape))
    V = np.zeros(shape)
    
    for idx,t in enumerate(self.__timeVec):
      V_ = self._var4IdTs(var_name, t, Z_idx)
      V[:,:,idx] = V_

    if t_firstaxis:
      V = np.moveaxis(V, -1, 0)

    return V


  def loadScalar(self, var_name):
    self.__vars[var_name] = self.varForId(var_name)


  def loadHorizontalVector(self, x_var_name, y_var_name, out_var_name):
    xVec = self.varForId(x_var_name)
    yVec = self.varForId(y_var_name)

    # Para face>6, los vectores (U,V) están en las coordenadas "locales"
    # Ver: https://github.com/MITgcm/MITgcm/issues/248 and https://github.com/MITgcm/xmitgcm/issues/204
    
    if self.__face>6:
       xVec,yVec = yVec,-1*xVec
    
    self.__vars[out_var_name] = (xVec, yVec)


  def get(self, var_name):
    return self.__vars[var_name]


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

