import os.path
import numpy as np
import multiprocessing
import luigi
import logging
import geopandas as gpd
from luigi import Task,LocalTarget,Parameter,IntParameter,FloatParameter,BoolParameter
from luigi.scheduler import Scheduler
from luigi.worker import Worker
from luigi.rpc import RemoteScheduler
from netCDF4 import Dataset
from xmitgcm import llcreader,open_mdsdataset
from scipy.interpolate import interp2d
##
from ..spectral_analysis.preprocess.common_vars import MODEL_FOLDER,DATA_FOLDER,LUIGI_OUT_FOLDER

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

# Directorio de los datos geográficos
prnt_map = "{}/map_data".format(DATA_FOLDER)
# Directorios de entrada
grid_path = "{}/LLC2160/grid/".format(MODEL_FOLDER)
# Directorios de salida
ds_path_fmt = LUIGI_OUT_FOLDER + "/Datasets_compressed/{}/{}"
filtered_path_fmt = LUIGI_OUT_FOLDER + "/Rob_k_filtered/{}/{}"

#min_i,min_j = 576,864	## Esquina inferior izquierda
#max_i,max_j = 865,1155	## Esquina superior derecha
#min_i,min_j = 576,577	## Esquina inferior izquierda
#max_i,max_j = 865,865	## Esquina superior derecha
#min_i,min_j = 576,288	## Esquina inferior izquierda
#max_i,max_j = 865,578	## Esquina superior derecha

Omega = 2*np.pi/(24*3600) # Frecuencia de rotación terrestre
vars_wf = ["U","V"] # Variables del modelo
#k_lvl_idx = [0, 36]  ## z=0 m, z=-400 m (39 para z=-500 m)
#k_lvl_suffixes = ["0","H"]
k_lvl_idx = [0]
k_lvl_suffixes = ["0"]

## Dias
# Tiempos correspondientes a los 91 días de JFM
idx_t_JFM_days = [i for i in range(377) if i>=110 and i<201]
# Tiempos correspondientes a los 92 días de ASO 2011-2012  * Antiguo
#idx_t_ASO = [i for i in range(377) if i>=323 or i<48]
# Tiempos correspondientes a los 85 días de JAS 2012
idx_t_JAS_days = [i for i in range(377) if i>=292]

## Horas
# Tiempos correspondientes a los 91*24 horas de JFM
idx_t_JFM_hours = [(24*i)+hr for i in range(377) if i>=110 and i<201 for hr in range(24)]
# Tiempos correspondientes a los 85*24-18 horas de JAS 2012. El 23 de septiembre solo tiene hasta las 5:00 h
idx_t_JAS_hours = [(24*i)+hr for i in range(377) if i>=292  for hr in range(24)][:-18]

idx_t = {
	"days": {
		"JFM": idx_t_JFM_days,
		"JAS": idx_t_JAS_days,
		"JFMJAS": idx_t_JFM_days+idx_t_JAS_days
	},
	"hours": {
		"JFM": idx_t_JFM_hours,
		"JAS": idx_t_JAS_hours,
		"JFMJAS": idx_t_JFM_hours+idx_t_JAS_hours
	}
}

area_latlonbox = {
	0: (-122.0,29.096750000000014,-116.0,33.827249999999985)
}

ids_Cal1 = [762, 787]			# test:0 California -- 23 to 51 N
ids_Cal2 = [809, 831, 852, 868] # .. (cont) Para las longitudes menores a 128
ids_Can = [709, 730, 750, 771] # Canarias -- 16 to 36 N
ids_Peru = [450, 572, 596, 616, 636] # Peru Chile -- 5 to 45 S
ids_Ben = [533, 556, 578, 602, 459] # Benguela -- 15 to 37 S ** Quitamos la **459** y ponemos la 602
ids_Kuro = [733, 751, 796]

faces_regions = {
	1: ids_Ben,
	2: ids_Can,
	7: ids_Kuro+ids_Cal2,
	10: ids_Cal1,
	11: ids_Peru
}

class GridMdsBase():
	time_prefix = Parameter()
	area_face = IntParameter()
	grid_ds = None
	#min_i,min_j,max_i,max_j = (None,)*4
	retry_count = 3

	def interpmat_48(self,grid_mat):
	    # Old
	    nrow,ncol = grid_mat.shape
	    rows = np.arange(nrow)
	    cols = np.arange(ncol)
	    interp = interp2d(cols,rows,grid_mat,kind="cubic")
	    # New 
	    new_r = np.arange(nrow*2)/2
	    new_c = np.arange(ncol*2)/2
	    return interp(new_c,new_r)

	def get_grid_ds(self):
		if self.grid_ds is None:
			logging.info("Reading grid Dataset")
			self.grid_ds = open_mdsdataset(grid_path, read_grid=True, iters=None, default_dtype=np.float32, geometry="llc").isel(face=self.area_face)
		return self.grid_ds

	def get_dxdy24(self):
		grid_ds = self.get_grid_ds()
		logging.info("Fetching grid dx")
		DX_24 = grid_ds.dxG.values
		logging.info("Fetching grid dy")
		DY_24 = grid_ds.dyG.values
		if self.area_face<=6:
			logging.info("Returning non-rotated version")
			return DX_24,DY_24
		else:
			logging.info("Returning rotated version")
			return np.rot90(DX_24),np.rot90(DY_24)

	def get_dxdy48(self):
		DX_24,DY_24 = self.get_dxdy24()
		logging.info("Interpolating grid")
		DX_48 = self.interpmat_48(DX_24/2)
		DY_48 = self.interpmat_48(DY_24/2)
		return DX_48,DY_48

	def get_lonlat24(self):
		grid_ds = self.get_grid_ds()
		logging.info("Fetching grid lon")
		LON_24 = grid_ds.XG.values
		logging.info("Fetching grid lat")
		LAT_24 = grid_ds.YG.values
		if self.area_face<=6:
			logging.info("Returning non-rotated version")
			return LON_24,LAT_24
		else:
			logging.info("Returning rotated version")
			return np.rot90(LON_24),np.rot90(LAT_24)

	def get_lonlat48(self):
		LON_24,LAT_24 = self.get_lonlat24()
		logging.info("Interpolating grid")
		LON_48 = self.interpmat_48(LON_24)
		LAT_48 = self.interpmat_48(LAT_24)
		return LON_48,LAT_48

	def read_indexes(self,area_id):
		logging.info("Reading indexes for area {}".format(area_id))
		with open("{}/lonlat_indexes.txt".format(ds_path_fmt.format(area_id,self.time_prefix)),'r') as f:
			idxx = tuple([int(x) for x in f.read().split(',')])
			logging.info("Indexes for area {}, ({})".format(area_id,idxx))
			return idxx

class SliceArea(Task,GridMdsBase):
	area_id = IntParameter()
	_target = None
	_target2 = None

	def find_ij_4lonlat(self,lon,lat,lon_m,lat_m):
	    # (lon,lat): Valores de [lon,lat] a buscar
	    # (lon_m,lat_m): Matrices con latitud y longitud en donde buscaremos
	    # Regresa: Índices i(lon),j(lat) correspondientes al punto más cercano a (lon_v,lat_v)
	    # Hacemos la búsqueda con el punto R2 = (lon_m-lon)^2 + (lat_m-lat)^2, tal que R2 sea mínimo
	    R2 = np.square(lon_m-lon) + np.square(lat_m-lat)
	    found = np.where(R2==np.amin(R2))
	    # Para xmitgcm, los índices están invertidos: i=eje x (columnas), j=eje y (filas)
	    j = found[0][0]
	    i = found[1][0]
	    return i,j

	def get_indexes(self):
		if self.area_id>0:
			geodata = gpd.read_file("{}/{}_{}_geo.json".format(prnt_map,"RV","ASO"),driver="GeoJSON")
			data_area = geodata[geodata["s_id"]==self.area_id].drop(columns=["season","var"])
			area_box = data_area["geometry"].values[0]
			lon_min,lat_min,lon_max,lat_max = area_box.bounds
			logging.info("Area {} box bounds (from geojson) {},{},{},{}:".format(self.area_id,lon_min,lat_min,lon_max,lat_max))
		else:
			lon_min,lat_min,lon_max,lat_max = area_latlonbox[self.area_id]
			logging.info("Area {} box bounds (from custom obj) {},{},{},{}:".format(self.area_id,lon_min,lat_min,lon_max,lat_max))
		
		#
		LON_48,LAT_48 = self.get_lonlat48()
		# Esquina inferior izquierda
		min_i,min_j = self.find_ij_4lonlat(lon_min,lat_min,LON_48,LAT_48)
		# Esquina superior derecha
		max_i,max_j = self.find_ij_4lonlat(lon_max,lat_max,LON_48,LAT_48)
		max_i = max_i+1
		max_j = max_j+1
		with self.output().open('w') as out:
			out.write("{},{},{},{}".format(min_i,min_j,max_i,max_j))
		with self.output2().open('w') as out2:
			out2.write("{},{},{},{}".format(lon_min,lat_min,lon_max,lat_max))

	def run(self):
		self.get_indexes()

	def output(self):
		if self._target is None:
			self._target = LocalTarget("{}/lonlat_indexes.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix)))
			self._target.makedirs()
		return self._target

	def output2(self):
		if self._target2 is None:
			self._target2 = LocalTarget("{}/lonlat_box.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix)))
			self._target2.makedirs()
		return self._target2

class GetGrid(Task,GridMdsBase):
	area_id = IntParameter()
	_target = None

	def requires(self):
		return SliceArea(area_id=self.area_id,time_prefix=self.time_prefix,area_face=self.area_face)

	def get_dxdy(self,min_j,max_j,min_i,max_i):
		logging.info("Getting dx,dy")
		DX_48,DY_48 = self.get_dxdy48()
		DX_48 = DX_48[min_j:max_j,min_i:max_i]
		DY_48 = DY_48[min_j:max_j,min_i:max_i]
		logging.info("Saving dx,dy")
		fname_dx = "{}/dx.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_dx,DX_48,fmt='%s')
		fname_dy = "{}/dy.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_dy,DY_48,fmt='%s')

	def get_lonlatf(self,min_j,max_j,min_i,max_i):
		logging.info("Getting lon,lat,f")
		LON_48,LAT_48 = self.get_lonlat48()
		logging.info("Lower left corner {},{}".format(LON_48[min_j,min_i],LAT_48[min_j,min_i]))
		logging.info("Upper right corner {},{}".format(LON_48[max_j,max_i],LAT_48[max_j,max_i]))
		LON_48 = LON_48[min_j:max_j,min_i:max_i]
		LAT_48 = LAT_48[min_j:max_j,min_i:max_i]
		sin_lat = np.sin(LAT_48*np.pi/180)
		FCOR_48 = 2*Omega*sin_lat

		logging.info("Saving lon,lat,f")
		fname_lon = "{}/lon.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_lon,LON_48,fmt='%s')
		fname_lat = "{}/lat.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_lat,LAT_48,fmt='%s')
		fname_lat = "{}/f.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_lat,FCOR_48,fmt='%s')
		

	def output(self):
		if self._target is None:
			self._target = [LocalTarget("{}/{}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),var_)) for var_ in ["dx","dy","lon","lat","f"]]
			self._target[0].makedirs()
		return self._target

	def run(self):
		min_i,min_j,max_i,max_j = self.read_indexes(self.area_id)
		self.get_dxdy(min_j,max_j,min_i,max_i)
		self.get_lonlatf(min_j,max_j,min_i,max_i)

class GetVelocities(Task,GridMdsBase):
	#area_id = IntParameter()
	#area_face = IntParameter()
	t = IntParameter()
	t_ds = None
	_target = None
	ds_area = None
	uv_vals = None

	def requires(self):
		for area_id in faces_regions[self.area_face]:
			yield SliceArea(area_id=area_id,time_prefix=self.time_prefix,area_face=self.area_face)

	def get_ds_area(self):
		if self.ds_area is None:
			#self.read_indexes(area_id)
			model = llcreader.ECCOPortalLLC4320Model()
			hourly_step = 24 if (self.time_prefix=="days") else 1
			logging.info("Class GetVelocities - reading variables ({}) Dataset (face={},t={},t_ds={})".format(vars_wf,self.area_face,self.t,self.t_ds))				# 144 iters = 1 hr, so 144*24 = 1 day
			self.ds_area = model.get_dataset(
				varnames=vars_wf,k_chunksize=len(k_lvl_idx),type="faces",iter_step=144*hourly_step
				).isel(time=self.t,face=self.area_face,k=k_lvl_idx)
			self.t_ds = self.ds_area.time.values
		return self.ds_area

	def clear_uv_vals(self):
		self.uv_vals = None

	def get_uv_vals_4_var_k(self,model_var,k_idx):
		# Solo si no se ha leido antes
		# Se da por hecho que después de iterar en las areas de cada cara, se corre self.clear_uv_vals()
		if self.uv_vals is None:
			logging.info("Class GetVelocities - reading {}{} (face={},t={})".format(model_var,k_lvl_suffixes[k_idx],self.area_face,self.t))
			ds_area_t_z = self.get_ds_area().isel(k=k_idx)
			self.uv_vals = ds_area_t_z[model_var].values
			# Corregimos para los recuadros del 7 en adelante
			if self.area_face>6:
				logging.info("We need to rotate this matrix as face={}>6".format(self.area_face))
				self.uv_vals = np.rot90(self.uv_vals)

	def save_uv_area(self,area_id,model_var,k_idx,k_suffix,min_j,max_j,min_i,max_i):
		fname_uv = "{}/{}{}_{:05d}.npz".format(ds_path_fmt.format(area_id,self.time_prefix),model_var,k_suffix,self.t)
		if os.path.exists(fname_uv) and os.path.getsize(fname_uv)>0:
			logging.info("Already exists. Skipping -- {}".format(fname_uv))
		else:
			# Leemos toda la cara
			self.get_uv_vals_4_var_k(model_var,k_idx)
			# Tomamos los datos al area correspondiente
			logging.info("Trimming {}{} data (face={},area_id={},t={},t_ds={}), i=({},{}), j=({},{})".format(model_var,k_suffix,self.area_face,area_id,self.t,self.t_ds,min_i,max_i,min_j,max_j))
			uv_vals_trimmed = self.uv_vals[min_j:max_j,min_i:max_i]	
			#np.savetxt(fname_uv,uv_vals_trimmed,fmt='%s')
			np.savez_compressed(fname_uv,uv=uv_vals_trimmed,t_ds=self.t_ds)

	def output(self):
		if self._target is None:
			self._target = [
				LocalTarget("{}/{}{}_{:05d}.npz".format(ds_path_fmt.format(area_id,self.time_prefix),var_,z_,self.t)) 
				for var_ in vars_wf for z_ in k_lvl_suffixes for area_id in faces_regions[self.area_face]
			]
			self._target[0].makedirs()
		return self._target

	def run(self):
		for var_name in vars_wf:
			for k_idx,k_suffix in enumerate(k_lvl_suffixes):
				for area_id in faces_regions[self.area_face]:
					logging.info("Loading indexes for area {} (face={},t={})".format(area_id,self.area_face,self.t))
					min_i,min_j,max_i,max_j = self.read_indexes(area_id)
					self.save_uv_area(area_id,var_name,k_idx,k_suffix,min_j,max_j,min_i,max_i)
				# Se tiene que limpiar los datos de la cara que se leyó, 
				# ya que save_uv_area() invoca get_uv_vals_4_var_k(),
				# el cual verifica que self.uv_vals sea None, de lo contrario no carga la cara siguiente
				self.clear_uv_vals()

class CreateNetCDF(Task,GridMdsBase):
	area_id = IntParameter()
	t = IntParameter()
	Lt_km = FloatParameter()

	lon = None
	lat = None
	dx = None
	dy = None
	f_cor = None
	Kx = None
	Ky = None
	filter_mask = None
	_target = None
	# Velocities (read and filtered)
	u0,v0 = None,None
	u0_lo,u0_hi,v0_lo,v0_hi = (None,)*4
	uH_lo,uH_hi,vH_lo,vH_hi = (None,)*4
	# RV
	RV_0,RV_H = None,None
	rv0_hi,rv0_lo,rvH_hi,rvH_lo = (None,)*4
	# OW
	OW_0,OW_H = None,None
	ow0_hi,ow0_lo,owH_hi,owH_lo = (None,)*4
	# Ro_b
	Ro_b,Rob_lo,Rob_hi = (None,)*3

	def init(self):
		if (self.dx is None) or (self.dy is None):
			name_dx = "{}/dx.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
			self.dx = np.loadtxt(name_dx)
			name_dy = "{}/dy.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
			self.dy = np.loadtxt(name_dy)
		if (self.f_cor is None) or (self.lat is None):
			# f_cor
			fname_f = "{}/f.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
			self.f_cor = np.loadtxt(fname_f)
			fname_lat = "{}/lat.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
			self.lat = np.loadtxt(fname_lat)
			
		if (self.Kx is None) or (self.Ky is None):
			# Kx,Ky
			fname_lon = "{}/lon.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
			self.lon = np.loadtxt(fname_lon)
			#logging.info("lon shape".format(self.lon.shape))
			Ny = self.lon.shape[0]
			Nx = self.lon.shape[1]
			Lx = Nx*np.mean(self.dx)
			Ly = Ny*np.mean(self.dy)
			delta_kx = 1/Lx
			delta_ky = 1/Ly
			N_kx = (Nx-1)/2
			N_ky = (Ny-1)/2
			self.Kx = delta_kx*np.arange(-N_kx,N_kx+1)
			self.Ky = delta_ky*np.arange(-N_ky,N_ky+1)

	def requires(self):
		return [GetGrid(area_id=self.area_id,time_prefix=self.time_prefix,area_face=self.area_face), GetVelocities(area_id=self.area_id,t=self.t,time_prefix=self.time_prefix,area_face=self.area_face)]

	def create_filter_k(order=0,flip=False): #Orden 0 es el filtro ideal, de 1 de adelante es el Butterworth
		# self.Kx y self.Ky están en metros, porque dx y dy lo están, así que Lt también debe estar en metros
		Lt_m = self.Lt_km*1000
		K_max_sq = (1/Lt)**2
		kx_,ky_ = np.meshgrid(self.Kx,self.Ky)

		if order==0: 	## "Ideal" filter
			filter_mask_plt = (np.square(kx_)+np.square(ky_) <= K_max_sq).astype(np.float32)
		else:			## Butterworth filter: http://fourier.eng.hmc.edu/e101/lectures/Fourier_Analysis/node10.html
			filter_mask_plt = 1/( 1 + ( (np.square(kx_)+np.square(ky_))/K_max_sq )**order )

		self.filter_mask = np.fft.fftshift(filter_mask_plt)

		if flip:
			self.filter_mask = self.filter_mask*np.fliplr(filter_mask) # Simetria en Kx
			self.filter_mask = self.filter_mask*np.flipud(filter_mask) # Simetria en Ky

	def filter_fft(self,var_xy):
		# FFT
		var_k = np.fft.fft2(var_xy)

		# Pasa bajas
		var_k_lo = var_k*self.filter_mask
		_var_lo = np.fft.ifft2(var_k_lo)
		var_lo = np.real(_var_lo)	# Eliminamos la parte imaginaria, ya que es espuria

		# Pasa altas lo tomaremos solamente como la diferencia entre el campo total (original) y el pasa bajas
		var_hi = var_xy-var_lo

		return var_lo,var_hi

	def filter_uv(self):
		logging.info("Partitioning spacial scales bigger and smaller than {:.2f} km".format(self.Lt_km))
		self.create_filter_k(order=100)
		## Z=0
		# Reading
		logging.info("U0")
		self.u0 = np.loadtxt("{}/U0_{:05d}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),self.t))
		logging.info("V0")
		self.v0 = np.loadtxt("{}/V0_{:05d}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),self.t))
		# Filtering
		logging.info ("Filter 0")
		self.u0_lo,self.u0_hi = self.filter_fft(self.u0)
		self.v0_lo,self.v0_hi = self.filter_fft(self.v0)

		## Z=H
		# Reading
		logging.info("UH")
		self.uH = np.loadtxt("{}/UH_{:05d}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),self.t))
		logging.info("VH")
		self.vH = np.loadtxt("{}/VH_{:05d}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),self.t))
		# Filtering
		logging.info("Filter H")
		self.uH_lo,self.uH_hi = self.filter_fft(self.uH)
		self.vH_lo,self.vH_hi = self.filter_fft(self.vH)

	def d_dx(self,f):
		return (np.gradient(f,axis=1,edge_order=2)/self.dx)

	def d_dy(self,f):
		return (np.gradient(f,axis=0,edge_order=2)/self.dy)

	def sn(self,u,v):
		return self.d_dx(u) - self.d_dy(v)

	def ss(self,u,v):
		return self.d_dx(v) + self.d_dy(u)

	def ow(self,u,v,rv):
		return np.square(self.sn(u,v)) + np.square(self.ss(u,v)) - np.square(rv)

	def rv(self,u,v):
		return self.d_dx(v) - self.d_dy(u)

	def rob_rv_st(self):
		## Z=0
		logging.info("RV-OW_0")
		self.RV_0 = self.rv(self.u0,self.v0)
		self.OW_0 = self.ow(self.u0,self.v0,self.RV_0)
		logging.info("RV-OW_0 Lo")
		self.rv0_lo = self.rv(self.u0_lo,self.v0_lo)
		self.ow0_lo = self.ow(self.u0_lo,self.v0_lo,self.rv0_lo)
		logging.info("RV-OW_0 Hi")
		self.rv0_hi = self.rv(self.u0_hi,self.v0_hi)
		self.ow0_hi = self.ow(self.u0_hi,self.v0_hi,self.rv0_hi)
		#  z=H
		logging.info("RV-OW_H")
		self.RV_H = self.rv(self.uH,self.vH)
		self.OW_H = self.ow(self.uH,self.vH,self.RV_H)
		logging.info("RV-OW_H Lo")
		self.rvH_lo = self.rv(self.uH_lo,self.vH_lo)
		self.owH_lo = self.ow(self.uH_lo,self.vH_lo,self.rvH_lo)
		logging.info("RV-OW_H Hi")
		self.rvH_hi = self.rv(self.uH_hi,self.vH_hi)
		self.owH_hi = self.ow(self.uH_hi,self.vH_hi,self.rvH_hi)

		logging.info("Ro_b")
		# ... y finalmente el Número de Rossby baroclínico (Ro_b)
		# ** Nota: No calculamos el valor absoluto para poder calcular promedios temporales,
		# 			en cuyo caso se toma abs() al promedio temporal
		self.Ro_b = (self.RV_0-self.RV_H)/self.f_cor
		self.Rob_lo = (self.rv0_lo-self.rvH_lo)/self.f_cor
		self.Rob_hi = (self.rv0_hi-self.rvH_hi)/self.f_cor

	def to_netcdf(self):
		fn_out = self.output().path
		logging.info("Saving to NetCDF: {}".format(fn_out))
		nc = Dataset(fn_out,'w',"NETCDF3")
		## Dims
		nlon,nlat=self.lon.shape
		nc.createDimension('nlon',nlon)
		nc.createDimension('nlat',nlat)
		nc.createDimension('k',2)
		nc.createDimension('time',None)   # Solo ponemos un tiempo
		## Coordenadas
		nlon_v = nc.createVariable('nlon','i4',('nlon'))
		nlon_v[:] = np.arange(nlon)
		nlat_v = nc.createVariable('nlat','i4',('nlat'))
		nlat_v[:] = np.arange(nlat)
		k_v = nc.createVariable('k','i4',('k'))
		k_v[:] = np.arange(2)
		## Atributos 
		# Lt en km
		nc.setncatts({'Lt_km':self.Lt_km})
		## Variables
		# Time
		time_v = nc.createVariable('time','i4',('time'))
		time_v.setncattr('units','{} since 2011-09-13T00:00:00'.format(self.time_prefix.title()))
		time_v[0] = self.t
		# Grid
		lon_v = nc.createVariable('lon','f4',('nlon','nlat'))
		lon_v.setncattr('units','degrees')
		lon_v[:,:] = self.lon
		lat_v = nc.createVariable('lat','f4',('nlon','nlat'))
		lat_v.setncattr('units','degrees')
		lat_v[:,:] = self.lat
		z_v = nc.createVariable('z','f4',('k'))
		z_v.setncatts({'units':'m','description':'Depth'})
		z_v[:] = [0,400]
		# Dx,dy
		dx_v = nc.createVariable('dx','f4',('nlon','nlat'))
		dx_v.setncatts({'units':'m','description':'dx per grid element'})
		dx_v[:,:] = self.dx
		dy_v = nc.createVariable('dy','f4',('nlon','nlat'))
		dy_v.setncatts({'units':'m','description':'dy per grid element'})
		dy_v[:,:] = self.dy
		# Coriolis
		fcor_v = nc.createVariable('f_coriolis','f4',('nlon','nlat'))
		fcor_v.setncatts({'units':'s^-1','description':'Coriolis parameter f'})
		fcor_v[:,:] = self.f_cor
		# Relative vorticity
		RV_v = nc.createVariable('RV','f4',('time','k','nlon','nlat'))
		RV_v.setncatts({'units':'s^-1','description':"Relative vorticity for all wavelenghts"})
		RV_v[0,0,:,:] = self.RV_0
		RV_v[0,1,:,:] = self.RV_H
		RV_lo_v = nc.createVariable('RV_Lo','f4',('time','k','nlon','nlat'))
		RV_lo_v.setncatts({'units':'s^-1','description':"Relative vorticity for wavelenghts > {:.1f} km".format(self.Lt_km)})
		RV_lo_v[0,0,:,:] = self.rv0_lo
		RV_lo_v[0,1,:,:] = self.rvH_lo
		RV_hi_v = nc.createVariable('RV_Hi','f4',('time','k','nlon','nlat'))
		RV_hi_v.setncatts({'units':'s^-1','description':"Relative vorticity for wavelenghts < {:.1f} km".format(self.Lt_km)})
		RV_hi_v[0,0,:,:] = self.rv0_hi
		RV_hi_v[0,1,:,:] = self.rvH_hi
		# Okubo-Weiss
		OW_v = nc.createVariable('OW','f4',('time','k','nlon','nlat'))
		OW_v.setncatts({'units':'s^-2','description':"Okubo-Weiss for all wavelenghts"})
		OW_v[0,0,:,:] = self.OW_0
		OW_v[0,1,:,:] = self.OW_H
		OW_lo_v = nc.createVariable('OW_Lo','f4',('time','k','nlon','nlat'))
		OW_lo_v.setncatts({'units':'s^-2','description':"Okubo-Weiss for wavelenghts > {:.1f} km".format(self.Lt_km)})
		OW_lo_v[0,0,:,:] = self.ow0_lo
		OW_lo_v[0,1,:,:] = self.owH_lo
		OW_hi_v = nc.createVariable('OW_Hi','f4',('time','k','nlon','nlat'))
		OW_hi_v.setncatts({'units':'s^-2','description':"Okubo-Weiss for wavelenghts < {:.1f} km".format(self.Lt_km)})
		OW_hi_v[0,0,:,:] = self.ow0_hi
		OW_hi_v[0,1,:,:] = self.owH_hi
		# Ro_b
		Rob_v = nc.createVariable('Rob','f4',('time','nlon','nlat'))
		Rob_v.setncatts({'units':'1','description':"Ro_b for all wavelenghts"})
		Rob_v[0,:,:] = self.Ro_b		
		Rob_lo_v = nc.createVariable('Rob_Lo','f4',('time','nlon','nlat'))
		Rob_lo_v.setncatts({'units':'1','description':"Ro_b for wavelenghts > {:.1f} km".format(self.Lt_km)})
		Rob_lo_v[0,:,:] = self.Rob_lo
		Rob_hi_v = nc.createVariable('Rob_Hi','f4',('time','nlon','nlat'))
		Rob_hi_v.setncatts({'units':'1','description':"Ro_b for wavelenghts < {:.1f} km".format(self.Lt_km)})
		Rob_hi_v[0,:,:] = self.Rob_hi

		nc.close()

	def output(self):
		if self._target is None:
			self._target = LocalTarget("{}/Rob_{:05d}.nc".format(filtered_path_fmt.format(self.area_id,self.time_prefix),self.t))
			self._target.makedirs()
		return self._target

	def run(self):
		Lt_m = self.Lt_km*1000
		self.init()
		self.filter_uv()
		self.rob_rv_st()
		self.to_netcdf()

class ProcessRegion(Task):
	region_id = IntParameter()
	Lt_km = FloatParameter()
	hourly = BoolParameter(default=False)

	def requires(self):
		t_pfx = "hours" if self.hourly else "days"
		steps = 9030 if self.hourly else 377
		for _t in range(steps):
			yield CreateNetCDF(area_id=self.region_id,Lt_km=self.Lt_km,t=_t,time_prefix=t_pfx,area_face=self.area_face)

class SliceAll(Task):
	t_pfx = "days"

	def requires(self):
		for area_face,ids in faces_regions.items():
			for area_id in ids:
				yield GetGrid(area_id=area_id,time_prefix=self.t_pfx,area_face=area_face)

## Use this https://stackoverflow.com/questions/21406887/subprocess-changing-directory/21406995
## and this https://stackoverflow.com/questions/36203059/execute-external-command-and-exchange-variable-using-python
## Concatenate all files, or by season
# ncrcat -n 377,3,1 Rob_00000.nc Rob_daily.nc
# ncrcat -n 91,3,1 Rob_00111.nc Rob_JFM.nc
# ncrcat -n 92,3,1,364,0 Rob_00322.nc Rob_ASO.nc
## Averaged files by season
# ncra -n 91,3,1 Rob_00111.nc Rob_JFM_avg.nc
# ncra -n 92,3,1,364,0 Rob_00322.nc Rob_ASO_avg.nc

class CustomWorkerFactory(object):
	def create_local_scheduler(self):
		return Scheduler(prune_on_get_work=True, record_task_history=False)

	def create_remote_scheduler(self, url):
		return RemoteScheduler(url)

	def create_worker(self, scheduler, worker_processes, assistant=False):
		# return your worker instance
		return Worker(scheduler=scheduler, worker_processes=worker_processes, assistant=assistant, keep_alive=True)


class DownloadVelocities(Task):
	time_prefix = Parameter()
	season = Parameter()

	def requires(self):
		for area_face,ids in faces_regions.items():
			for area_id in ids:
				yield GetGrid(area_id=area_id,area_face=area_face,time_prefix=self.time_prefix)
			for t in idx_t[self.time_prefix][self.season]:
				yield GetVelocities(t=t,area_face=area_face,time_prefix=self.time_prefix)


if __name__ == "__main__":
	logging.info("Starting Luigi tasks")
	#n_workers = multiprocessing.cpu_count()
	n_workers = 5
	wf_result = luigi.build([DownloadVelocities(time_prefix="hours",season="JFMJAS")], workers=n_workers, detailed_summary=True)
	#wf_result = luigi.build([SliceAll()], workers=n_workers, detailed_summary=True)
	
	#wf_result = luigi.build([ProcessRegion(region_id=771,Lt_km=75)], workers=n_workers, detailed_summary=True)

	#wf_result = luigi.build([ProcessRegion(region_id=730,Lt_km=35)], workers=n_workers, detailed_summary=True)

