import logging
import numpy as np


## Directorios
LUIGI_OUT_FOLDER = "/home/antonio/Tesis"	## Ponerla en common_vars.directories hace más complicado su importación
ds_path_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}"
dxdy_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{}/{}/{}.txt"
uv_fname_fmt = LUIGI_OUT_FOLDER+"/Datasets_compressed/{0}/{1}/{2}{3:02d}_{4:05d}.npz"


def uv4idt(region_id,t_idx,Z_idx,t_res="hours"):
	fname_u = uv_fname_fmt.format(region_id,t_res,"U",Z_idx,t_idx)
	fname_v = uv_fname_fmt.format(region_id,t_res,"V",Z_idx,t_idx)
	U_ = np.load(fname_u)["uv"]
	V_ = np.load(fname_v)["uv"]
	return U_,V_


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

