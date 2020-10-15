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
# GCP profiling
import googlecloudprofiler
# Imports within the same package
from ..common_vars.directories import LUIGI_OUT_FOLDER
from ..common_vars.time_slices import max_iter,idx_t
from ..common_vars.regions import faces_regions
from ..common_vars.directories import MODEL_FOLDER,DATA_FOLDER

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

# Directorio de los datos geográficos
prnt_map = "{}/map_data".format(DATA_FOLDER)
# Directorio de entrada
grid_path = "{}/LLC4320/grid/".format(MODEL_FOLDER)
# Directorio de salida
ds_path_fmt = LUIGI_OUT_FOLDER + "/Datasets_compressed/{}/{}"

Omega = 2*np.pi/(24*3600) # Frecuencia de rotación terrestre
vars_wf = ["U","V"] # Variables del modelo
#vars_wf = ["oceQnet","oceTAUX","oceTAUY"]
#vars_wf = ["oceQnet"]
#vars_wf = ["Eta"]
#k_lvl_idx = [0, 36]  ## z=0 m, z=-400 m (39 para z=-500 m)
#k_lvl_idx = [0,6,12,16,19]#,21,25 ## Para T, y quizás S
k_lvl_idx = [0]

## For test purposes only
area_latlonbox = {
	0: (-122.0,29.096750000000014,-116.0,33.827249999999985)
}

class GridMdsBase():
	time_prefix = Parameter()
	area_face = IntParameter()
	grid_ds = None
	#min_i,min_j,max_i,max_j = (None,)*4
	retry_count = 3

	def get_grid_ds(self):
		if self.grid_ds is None:
			logging.info("Reading grid Dataset")
			self.grid_ds = open_mdsdataset(grid_path, read_grid=True, iters=None, default_dtype=np.float32, geometry="llc").isel(face=self.area_face)
		return self.grid_ds

	def get_grid48(self):
		grid_ds = self.get_grid_ds()
		logging.info("Fetching grid DX_g")
		DXg = grid_ds.dxG.values
		logging.info("Fetching grid DX_c")
		DXc = grid_ds.dxC.values
		logging.info("Fetching grid DY_g")
		DYg = grid_ds.dyG.values
		logging.info("Fetching grid DY_c")
		DYc = grid_ds.dyC.values
		logging.info("Fetching grid rA_z")
		rAz = grid_ds.rAz.values
		logging.info("Fetching grid rA_c")
		rAc = grid_ds.rA.values

		if self.area_face<=6:
			logging.info("Returning non-rotated version")
			return DXg,DYg,rAz,DXc,DYc,rAc
		else:
			logging.info("Returning rotated version")
			return np.rot90(DXg),np.rot90(DYg),np.rot90(rAz),np.rot90(DXc),np.rot90(DYc),np.rot90(rAc)


	def get_lonlat48(self):
		grid_ds = self.get_grid_ds()
		logging.info("Fetching grid lon_g")
		LONg = grid_ds.XG.values
		logging.info("Fetching grid lat_g")
		LATg = grid_ds.YG.values
		logging.info("Fetching grid lon_c")
		LONc = grid_ds.XC.values
		logging.info("Fetching grid lat_c")
		LATc = grid_ds.YC.values

		if self.area_face<=6:
			logging.info("Returning non-rotated version")
			return LONg,LATg,LONc,LATc
		else:
			logging.info("Returning rotated version")
			return np.rot90(LONg),np.rot90(LATg),np.rot90(LONc),np.rot90(LATc)

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
		LONg,LATg,LONc,LATc = self.get_lonlat48()
		# Esquina inferior izquierda
		min_i,min_j = self.find_ij_4lonlat(lon_min,lat_min,LONg,LATg)
		# Esquina superior derecha
		max_i,max_j = self.find_ij_4lonlat(lon_max,lat_max,LONg,LATg)
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
		DXg,DYg,rAz,DXc,DYc,rAc = self.get_grid48()
		DXg = DXg[min_j:max_j,min_i:max_i]
		DYg = DYg[min_j:max_j,min_i:max_i]
		rAz = rAz[min_j:max_j,min_i:max_i]
		DXc = DXc[min_j:max_j,min_i:max_i]
		DYc = DYc[min_j:max_j,min_i:max_i]
		rAc = rAc[min_j:max_j,min_i:max_i]

		logging.info("Saving dx,dy,rA")
		fname_ = "{}/dxg.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,DXg,fmt='%s')
		fname_ = "{}/dyg.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,DYg,fmt='%s')
		fname_ = "{}/rAz.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,rAz,fmt='%s')
		fname_ = "{}/dxc.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,DXc,fmt='%s')
		fname_ = "{}/dyc.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,DYc,fmt='%s')
		fname_ = "{}/rAc.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,rAc,fmt='%s')

	def get_lonlatf(self,min_j,max_j,min_i,max_i):
		logging.info("Getting lon,lat,f")
		LONg,LATg,LONc,LATc = self.get_lonlat48()
		# Las coordenadas originales (de lonlat_box) deberían ser más parecidas a las (g) -- Ver SliceArea.get_indexes()
		logging.info("Lower left corner (g) {},{}".format(LONg[min_j,min_i],LATg[min_j,min_i]))
		logging.info("Upper right corner  (g) {},{}".format(LONg[max_j,max_i],LATg[max_j,max_i]))
		logging.info("Lower left corner (g) {},{}".format(LONc[min_j,min_i],LATc[min_j,min_i]))
		logging.info("Upper right corner  (g) {},{}".format(LONc[max_j,max_i],LATc[max_j,max_i]))
		LONg = LONg[min_j:max_j,min_i:max_i]
		LATg = LATg[min_j:max_j,min_i:max_i]
		LONc = LONc[min_j:max_j,min_i:max_i]
		LATc = LATc[min_j:max_j,min_i:max_i]
		sin_lat = np.sin(LATc*np.pi/180)
		FCOR_48 = 2*Omega*sin_lat

		logging.info("Saving lon,lat,f")
		fname_ = "{}/lon_g.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,LONg,fmt='%s')
		fname_ = "{}/lat_g.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,LATg,fmt='%s')
		fname_ = "{}/lon_c.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,LONc,fmt='%s')
		fname_ = "{}/lat_c.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_,LATc,fmt='%s')
		fname_f = "{}/f.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix))
		np.savetxt(fname_f,FCOR_48,fmt='%s')
		

	def output(self):
		if self._target is None:
			self._target = [LocalTarget("{}/{}.txt".format(ds_path_fmt.format(self.area_id,self.time_prefix),var_)) \
			 for var_ in ["dxg","dyg","rAz","lon_g","lat_g","dxc","dyc","rAc","lon_c","lat_c","f"]]
			self._target[0].makedirs()
		return self._target

	def run(self):
		min_i,min_j,max_i,max_j = self.read_indexes(self.area_id)
		self.get_dxdy(min_j,max_j,min_i,max_i)
		self.get_lonlatf(min_j,max_j,min_i,max_i)


class GetSingleVariable(Task,GridMdsBase):
	t = IntParameter()
	model_var = Parameter()
	t_ds = None
	_target = None
	ds_area = None
	values = None

	def requires(self):
		for area_id in faces_regions[self.area_face]:
			yield SliceArea(area_id=area_id,time_prefix=self.time_prefix,area_face=self.area_face)

	def get_ds_area(self):
		if self.ds_area is None:
			#self.read_indexes(area_id)
			model = llcreader.ECCOPortalLLC4320Model()
			model.iter_stop = max_iter
			hourly_step = 24 if (self.time_prefix=="days") else 1
			self.ds_area = model.get_dataset(
				varnames=[self.model_var],k_chunksize=1,type="faces",iter_step=144*hourly_step
				).isel(time=self.t,face=self.area_face,k=k_lvl_idx)
			self.t_ds = self.ds_area.time.values
			logging.info("Class GetSingleVariable - reading variables ({}) Dataset (face={},t={},t_ds={})".format(vars_wf,self.area_face,self.t,self.t_ds))				# 144 iters = 1 hr, so 144*24 = 1 day
		return self.ds_area

	def clear_values(self):
		self.values = None

	def get_values_4_var_k(self,model_var,k_idx,k_lvl):
		# Solo si no se ha leido antes
		# Se da por hecho que después de iterar en las areas de cada cara, se corre self.clear_values()
		if self.values is None:
			logging.info("Class GetSingleVariable - reading {}{:02d} (face={},t={})".format(model_var,k_lvl,self.area_face,self.t))
			ds_area_t_z = self.get_ds_area().isel(k=k_idx)
			self.values = ds_area_t_z[model_var].values
			# Corregimos para los recuadros del 7 en adelante
			if self.area_face>6:
				logging.info("We need to rotate this matrix as face={}>6".format(self.area_face))
				self.values = np.rot90(self.values)

	def save_uv_area(self,area_id,model_var,k_idx,k_lvl,min_j,max_j,min_i,max_i):
		fname_uv = "{}/{}{:02d}_{:05d}.npz".format(ds_path_fmt.format(area_id,self.time_prefix),model_var,k_lvl,self.t)
		if os.path.exists(fname_uv) and os.path.getsize(fname_uv)>0:
			logging.info("Already exists. Skipping -- {}".format(fname_uv))
		else:
			# Leemos toda la cara
			self.get_values_4_var_k(model_var,k_idx,k_lvl)
			# Tomamos los datos al area correspondiente
			logging.info("Trimming {}{:02d} data (face={},area_id={},t={},t_ds={}), i=({},{}), j=({},{})".format(model_var,k_lvl,self.area_face,area_id,self.t,self.t_ds,min_i,max_i,min_j,max_j))
			values_trimmed = self.values[min_j:max_j,min_i:max_i]	
			#np.savetxt(fname_uv,values_trimmed,fmt='%s')
			np.savez_compressed(fname_uv,uv=values_trimmed,t_ds=self.t_ds)
			logging.info("Saved: {}".format(fname_uv))

	def output(self):
		if self._target is None:
			self._target = [
				LocalTarget("{}/{}{:02d}_{:05d}.npz".format(ds_path_fmt.format(area_id,self.time_prefix),self.model_var,z_,self.t)) 
				for z_ in k_lvl_idx for area_id in faces_regions[self.area_face]
			]
			self._target[0].makedirs()
		return self._target

	def run(self):
		for k_idx,k_lvl in enumerate(k_lvl_idx):
			for area_id in faces_regions[self.area_face]:
				logging.info("Loading indexes for area {} (face={},t={})".format(area_id,self.area_face,self.t))
				min_i,min_j,max_i,max_j = self.read_indexes(area_id)
				self.save_uv_area(area_id,self.model_var,k_idx,k_lvl,min_j,max_j,min_i,max_i)
				# Se tiene que limpiar los datos de la cara que se leyó, 
				# ya que save_uv_area() invoca get_values_4_var_k(),
				# el cual verifica que self.values sea None, de lo contrario no carga la cara siguiente
				self.clear_values()

class SliceAll(Task):
	t_pfx = "days"

	def requires(self):
		for area_face,ids in faces_regions.items():
			for area_id in ids:
				yield GetGrid(area_id=area_id,time_prefix=self.t_pfx,area_face=area_face)

class GetGrids(Task):
	time_prefix = Parameter()

	def requires(self):
		for area_face,ids in faces_regions.items():
			for area_id in ids:
				yield GetGrid(area_id=area_id,area_face=area_face,time_prefix=self.time_prefix)


class DownloadVariables(Task):
	time_prefix = Parameter()
	season = Parameter()

	def requires(self):
		for area_face,ids in faces_regions.items():
			for area_id in ids:
				yield GetGrid(area_id=area_id,area_face=area_face,time_prefix=self.time_prefix)
			for t in idx_t[self.time_prefix][self.season]:
				for var_ in vars_wf:
					yield GetSingleVariable(t=t,model_var=var_,area_face=area_face,time_prefix=self.time_prefix)


# Renamed with 
#	find . -type f -wholename "*/[UV]H_*.npz" -execdir rename -n 's/\.\/(.+)H_(.+)/${1}36_${2}/' {} \;^C
#	find . -type f -wholename "*/[UV]0_*.npz" -execdir rename -n 's/\.\/(.+)0_(.+)/${1}36_${2}/' {} \;^C

# Cleaned up grid files with
# find . -wholename *hours/lon.txt -delete
# find . -wholename *hours/lat.txt -delete
# find . -wholename *hours/dx.txt -delete
# find . -wholename *hours/dy.txt -delete
# find . -wholename *days/lon.txt -delete
# find . -wholename *days/lat.txt -delete
# find . -wholename *days/dx.txt -delete
# find . -wholename *days/dy.txt -delete

#n_workers = multiprocessing.cpu_count()

luigi_opts = {
    "workers": 10,
    "detailed_summary": False,
    "scheduler_host": "10.138.0.2", 
    "scheduler_port": 8080
}

if __name__ == "__main__":
	logging.info("Starting Luigi tasks")	

	#wf_result = luigi.build([SliceAll()], **luigi_opts)

	#wf_result = luigi.build([GetGrids(time_prefix="hours")], **luigi_opts)
	try:
		googlecloudprofiler.start(
			service='luigi-worker-profiler',
			service_version='1.5.0',
			# verbose is the logging level. 0-error, 1-warning, 2-info,
			# 3-debug. It defaults to 0 (error) if not set.
			verbose=3,
			# project_id must be set if not running on GCP.
			# project_id='my-project-id',
		)
	except (ValueError, NotImplementedError) as exc:
		print(exc)  # Handle errors here

	wf_result = luigi.build([DownloadVariables(time_prefix="hours",season="JFM")], **luigi_opts)
