import logging
import numpy as np
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


  def __init__(self, rid, timeVec, t_res="hours"):
    self.__vars = {}
    self.__regionId = rid
    self.__timeRes = t_res
    self.__timeVec = timeVec
    self.__grid = VorticityGrid(rid, t_res)

  def _var4IdTs(self, var_name, t_idx, Z_idx):
    fname_hf = uv_fname_fmt.format(self.__regionId, self.__timeRes, var_name, Z_idx, t_idx)
    v = np.load(fname_hf)["uv"]

    return v


  def varForId(self, var_name, Z_idx=0, t_firstaxis=False):
    for idx,t in enumerate(self.__timeVec):
      V_ = self._var4IdTs(var_name, t, Z_idx)
      if idx==0:
        logging.debug("{},{}".format(id, t))
        shape_uv = V_.shape
        shape = (shape_uv[0], shape_uv[1], len(self.__timeVec))
        logging.info("Theta shape (k={}): {}".format(Z_idx, shape))
        V = np.zeros(shape)
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
    face = face4id[self.__regionId]
    if face>6:
       xVec,yVec = yVec,-1*xVec
    
    self.__vars[out_var_name] = (xVec, yVec)

  def norm(self, vec_var_name, out_var_name):
    xVec,yVec = self.__vars[vec_var_name]
    self.__vars[out_var_name] = np.sqrt(xVec**2 + yVec**2)

  def divergence(self, vec_var_name, out_var_name):
    xVec,yVec = self.__vars[vec_var_name]
    div = self.__grid.div(np.moveaxis(xVec,-1,0), np.moveaxis(yVec,-1,0))
    self.__vars[out_var_name] = np.moveaxis(div, 0, -1)

  def hcurl(self, vec_var_name, out_var_name):
    xVec,yVec = self.__vars[vec_var_name]
    rot = self.__grid.rv(np.moveaxis(xVec,-1,0), np.moveaxis(yVec,-1,0))
    self.__vars[out_var_name] = np.moveaxis(rot, 0, -1)


  def get(self, var_name):
    return self.__vars[var_name]

