from pandas import read_csv
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# Imports within the same package
from ..tools.utils_thesis import coriolis

#client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
#pritn(client)

N = 250 # Cada cuantos elementos vamos submuestrear --  evitamos leer todo el dataset, ya que no es necesario
etopo_fn = 'ETOPO1/ETOPO1_Bed_g_gmt4.grd'

print("Opening ETOPO1 dataset")
xds = xr.open_dataset(etopo_fn)
print("Exporting to DF")
df = xds.to_dataframe()[::N] # Este DF tiene como indices las posiciones 'x' (lon) e 'y'(lat). 'z'

x = df.index.get_level_values(0)
y = df.index.get_level_values(1)
#z = df['z']

print("Creating GeoPandas DF")
etopo_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y))

lon = etopo_gdf.index.get_level_values(0)
lat = etopo_gdf.index.get_level_values(1)
z = etopo_gdf['z']
etopo_gdf['H'] = -z # Revertimos el signo, ya que necesitamos la profundidad H

#https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
#print("Displaying bathymetry scatterplot")
#sc = plt.scatter(lon,lat,c=z,s=0.25)
#plt.colorbar(sc)
#plt.show()

#
print("Reading data c1 y Rd1 (Chelton et al., 1998)")
rossby_cRd_folder = 'chelton__baroclinic_c_Rd'
rossby_cRd = read_csv("{}/rossrad.dat".format(rossby_cRd_folder),sep=' +',names=['lat_cRd','lon_cRd','c1','Rd1'])
rossby_cRd["lon_cRd"] = rossby_cRd["lon_cRd"].map(lambda x: x if x<180 else x-360) # Corregimos la longitud.

print("Creating GeoPandas DF")
rossby_cRd = gpd.GeoDataFrame(rossby_cRd, geometry=gpd.points_from_xy(rossby_cRd.lon_cRd, rossby_cRd.lat_cRd))
#print("shape:{}, min_lon: {}, max_lon: {}".format(rossby_cRd.shape,rossby_cRd["lon_cRd"].min(),rossby_cRd["lon_cRd"].max()))
rossby_cRd.head()

#print("Displaying Rd1 scatterplot")
x = rossby_cRd["lon_cRd"]
y = rossby_cRd["lat_cRd"]
s = rossby_cRd["Rd1"]
#plt.scatter(x,y,c=s,s=0.25)
#plt.show()

#
poly_fn = 'map_data/KE_ASO_geo.json'
print("Getting polygons from {}".format(poly_fn))
poly_cols = ['lat','lon','s_id','geometry'] # We only need id and geometry
poly_gdf = gpd.read_file(poly_fn,driver='GeoJSON').set_index('s_id',drop=False)[poly_cols]
poly_gdf['f_cph'] = coriolis(poly_gdf['lat'])

print("Combining polygons with Chelton (c1,Rd1) data")
geodata_cRd = gpd.sjoin(poly_gdf,rossby_cRd,how='inner',op='intersects')
#print(geodata_cRd.shape)
#geodata_cRd.head()
print("Calculating avg c1,Rd1 per polygon")
geodata_cRd = geodata_cRd.groupby(["s_id"]).agg({"c1": np.mean, "Rd1": np.mean})
geodata_cRd['s_id'] = geodata_cRd.index.values
geodata_cRd.head()

print("Combining result with ETOPO1 data")
geodata_all = gpd.sjoin(poly_gdf,etopo_gdf.reset_index(),how='inner',op='intersects') # reset_index elimina los indices x,y para volverlo simple (secuencial)
#print(geodata_all.shape)
print("Calculating avg depth per polygon")
geodata_all = geodata_all.groupby(["s_id"]).agg({"H": np.mean, "f_cph": np.mean})

print("Combining all results")
geodata_all = geodata_cRd.join(geodata_all,how='inner')

print("Calculating buoyancy frequency: N = (pi*c1)/H, and N_2 = f*Rd/H")
geodata_all['Nbv_rad_s'] = np.pi*np.divide(geodata_all['c1'],geodata_all['H'])
geodata_all['Nbv_cph'] = (3600/(2*np.pi))*geodata_all['Nbv_rad_s']
geodata_all['Nbv_cph_2'] = np.pi*np.divide(np.multiply(np.abs(geodata_all['f_cph']),1000*geodata_all['Rd1']),geodata_all['H'])

geodata_all.head()

out_fn = 'merged_Rd_c1_H_Nbv.csv'
print("Saving as CSV: {}".format(out_fn))
geodata_all.to_csv(out_fn)


