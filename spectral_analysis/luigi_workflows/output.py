import logging
import numpy as np
#
from ..common_vars.directories import LUIGI_OUT_FOLDER
from ..common_vars.regions import face4id

## Directorios
ds_path_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}"
dxdy_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}/{}.txt"
uv_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{0}/{1}/{2}{3:02d}_{4:05d}.npz"

def theta4idt(region_id,t_idx,Z_idx,t_res="hours"):
	fname_hf = uv_fname_fmt.format(region_id,t_res,"Theta",Z_idx,t_idx)
	hf = np.load(fname_hf)["uv"]
	
	return hf

def Theta4id(id,time,Z_idx=0,t_res="hours",t_firstaxis=False):
	for idx,t in enumerate(time):
		T_ = theta4idt(id,t,Z_idx,t_res)
		if idx==0:
			logging.debug("{},{}".format(id,t))
			shape_uv = T_.shape
			shape = (shape_uv[0],shape_uv[1],len(time))
			logging.info("Theta shape (k={}): {}".format(Z_idx,shape))
			T = np.zeros(shape)
		T[:,:,idx] = T_

	if t_firstaxis:
		T = np.moveaxis(T,-1,0)

	return T

def heatFlux4idt(region_id,t_idx,Z_idx,t_res="hours"):
	fname_hf = uv_fname_fmt.format(region_id,t_res,"oceQnet",Z_idx,t_idx)
	hf = np.load(fname_hf)["uv"]
	
	return hf

def H4id(id,time,Z_idx=0,t_res="hours",t_firstaxis=False):
	for idx,t in enumerate(time):
		HF_ = heatFlux4idt(id,t,Z_idx,t_res)
		if idx==0:
			logging.debug("{},{}".format(id,t))
			shape_uv = HF_.shape
			shape = (shape_uv[0],shape_uv[1],len(time))
			logging.info("HF shape (k={}): {}".format(Z_idx,shape))
			HF = np.zeros(shape)
		HF[:,:,idx] = HF_

	if t_firstaxis:
		HF = np.moveaxis(HF,-1,0)

	return HF


def uv4idt(region_id,t_idx,Z_idx,t_res="hours"):
	fname_u = uv_fname_fmt.format(region_id,t_res,"U",Z_idx,t_idx)
	fname_v = uv_fname_fmt.format(region_id,t_res,"V",Z_idx,t_idx)
	U_ = np.load(fname_u)["uv"]
	V_ = np.load(fname_v)["uv"]

	face = face4id[region_id]
	if face<=6:
		return U_,V_
	# Para face>6, los vectores (U,V) están en las coordenadas "locales"
	# Ver: https://github.com/MITgcm/MITgcm/issues/248 and https://github.com/MITgcm/xmitgcm/issues/204
	else:
		return V_,-1*U_


def UV4id(id,time,Z_idx=0,t_res="hours",t_firstaxis=False):
	for idx,t in enumerate(time):
		U_,V_ = uv4idt(id,t,Z_idx,t_res)
		if idx==0:
			logging.debug("{},{}".format(id,t))
			shape_uv = U_.shape
			shape = (shape_uv[0],shape_uv[1],len(time))
			logging.info("UV shape (k={}): {}".format(Z_idx,shape))
			U = np.zeros(shape)
			V = np.zeros(shape)
		U[:,:,idx] = U_
		V[:,:,idx] = V_

	if t_firstaxis:
		U = np.moveaxis(U,-1,0)
		V = np.moveaxis(V,-1,0)

	return U,V

def tau4idt(region_id,t_idx,Z_idx,t_res="hours"):
	fname_taux = uv_fname_fmt.format(region_id,t_res,"oceTAUX",Z_idx,t_idx)
	fname_tauy = uv_fname_fmt.format(region_id,t_res,"oceTAUY",Z_idx,t_idx)
	Tx_ = np.load(fname_taux)["uv"]
	Ty_ = np.load(fname_tauy)["uv"]

	face = face4id[region_id]
	if face<=6:
		return Tx_,Ty_
	# Para face>6, los vectores (tauX,tauY) están en las coordenadas "locales"
	# Ver: https://github.com/MITgcm/MITgcm/issues/248 and https://github.com/MITgcm/xmitgcm/issues/204
	else:
		return Ty_,-1*Tx_


def Tau4id(id,time,Z_idx=0,t_res="hours",t_firstaxis=False):
	for idx,t in enumerate(time):
		Tx_,Ty_ = tau4idt(id,t,Z_idx,t_res)
		if idx==0:
			logging.debug("{},{}".format(id,t))
			shape_uv = Tx_.shape
			shape = (shape_uv[0],shape_uv[1],len(time))
			logging.info("TAUxy shape (k={}): {}".format(Z_idx,shape))
			TauX = np.zeros(shape)
			TauY = np.zeros(shape)
		TauX[:,:,idx] = Tx_
		TauY[:,:,idx] = Ty_

	if t_firstaxis:
		TauX = np.moveaxis(TauX,-1,0)
		TauY = np.moveaxis(TauY,-1,0)

	return TauX,TauY


class VorticityGrid():
	dxc,dxg,dyc,dyg,rAz,rAc,lon_c,lon_g,lat_c,lat_g,f = (None,)*11
	
	def __init__(self,region_id,t_res="hours"):
		self.dxg = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"dxg"))
		self.dxc = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"dxc"))
		self.dyg = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"dyg"))
		self.dyc = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"dyc"))
		self.rAz = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"rAz"))
		self.rAc = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"rAc"))
		self.lon_c = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"lon_c"))
		self.lon_g = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"lon_g"))
		self.lat_c = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"lat_c"))
		self.lat_g = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"lat_g"))
		self.f = np.loadtxt(dxdy_fname_fmt.format(region_id,t_res,"f"))
	
	def xyg(self):
		return self.lon_g,self.lat_g
	
	def xyc(self):
		return self.lon_c,self.lat_c
	
	## Ejes para derivadas (pensando en una matriz 2D, las columnas van en X y las filas en Y)
	# Axis -1: x (último, o columnas (1) si es 2D)
	# Axis -2: y (penúltimo, o filas (0) si es 2D)
	# Si se agrega el tiempo, éste debe ser la primera dimensión
	
	## http://mitgcm.org/sealion/online_documents/node61.html
	# Centrado en celdas g (lat_g,lon_g)
	def rv(self,U,V):
		return (np.gradient(self.dyc*V,axis=-1,edge_order=2) - np.gradient(self.dxc*U,axis=-2,edge_order=2))/self.rAz
	
	def rv2(self,U,V):
		return np.square(self.rv(U,V))

	def st(self,U,V):
		return np.sqrt(self.st2(U,V))

	def st2(self,U,V):
		ss = (np.gradient(self.dyc*V,axis=-1,edge_order=2) + np.gradient(self.dxc*U,axis=-2,edge_order=2))/self.rAz
		sn = (np.gradient(self.dyc*U,axis=-1,edge_order=2) - np.gradient(self.dxc*V,axis=-2,edge_order=2))/self.rAz
		return np.square(sn) + np.square(ss)

	def ow(self,U,V):
		return self.st2(U,V) - self.rv2(U,V)

	## http://mitgcm.org/sealion/online_documents/node43.html
	# Centrado en celdas c (lat_c,lon_c)
	def div(self,U,V):
		return (np.gradient(self.dyg*U,axis=-1,edge_order=2) + np.gradient(self.dxg*V,axis=-2,edge_order=2))/self.rAc

