import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from xmitgcm import llcreader,open_mdsdataset
#
from ..common_vars.directories import MODEL_FOLDER

## Importing model data
model = llcreader.ECCOPortalLLC4320Model()
# Parámetros para la extracción de los datos
map_mode = 'faces' # 'faces' or 'latlon'
vars_wf = ["U","V","Theta"] # Variables del modelo
ds = model.get_dataset(varnames=vars_wf,k_chunksize=1,type=map_mode,iter_step=144*24) # 144 iters = 1 hr

## Importing model grid
grid_path = "{}/LLC2160/grid/".format(MODEL_FOLDER)
grid_ds = open_mdsdataset(grid_path, read_grid=True, iters=None, default_dtype=np.float32, geometry='llc')


## Interpolating grid (from 1/24 to 1/48)
# Reading
j24 = grid_ds.j.values
i24 = grid_ds.i.values
XG_24 = xr.DataArray(grid_ds.XG.values,[('face',grid_ds.face.values),('j',2*j24),('i',2*i24)])
YG_24 = xr.DataArray(grid_ds.YG.values,[('face',grid_ds.face.values),('j',2*j24),('i',2*i24)])
# Interpolating
j48 = [jj for jj in range(2*j24[-1]+2)]
i48 = [ii for ii in range(2*i24[-1]+2)]
XG_48 = XG_24.interp(j=j48,i=i48,kwargs={'fill_value':None})
YG_48 = YG_24.interp(j=j48,i=i48,kwargs={'fill_value':None})
# Creating dataset from interpolated grid
coords_48 = xr.Dataset({'lon':XG_48,'lat':YG_48},{'face':grid_ds.face.values,'j':j48,'i':i48})


## Función que obtiene los datos de una variable para una región dada por los límites [(min_lon,min_lat),(max_lon,max_lat)]
## Limitaciones: No se puede elegir un rango (slice) de tiempo("time") ni de profundidad ("k")
def readvar_from_latlonbox(min_lon,min_lat,max_lon,max_lat,model_var,time=0,k=0):
    # No se aceptará otra variable que no esté disponible
    assert model_var in vars_wf, "model_var should be one of {}, but is {}".format(vars_wf,model_var)
    # Validamos que las coordenadas sean válidas y que al menos se pueda tomar un punto de la malla
    assert (min_lon>=-180 and min_lon<=180) and (max_lon>=-180 and max_lon<=180) and (min_lat>=-90 and min_lat<=90) and (max_lat>=-90 and max_lat<=90), "lon are out of range ([-180,180],[-90,90])"
    assert max_lon-min_lon>1/48, "min_lon should be less than max_lon and at least 1/48° apart"
    assert max_lat-min_lat>1/48, "min_lat should be less than max_lat and at least 1/48° apart"

    # Esta máscara representa la caja que usaremos
    mask = (XG_48 > min_lon) & (XG_48 < max_lon) & (YG_48 > min_lat) & (YG_48 < max_lat)
    # Pre-seleccionamos los datos -- en realidad sirve más para saber en qué caras (faces) están los datos que buscamos (por eso el "pre")
    vals2plot = ds[model_var].isel(time=time,k=k).where(mask, drop=True)
# ... y las coordenadas
    cords2plot = coords_48.isel(face=vals2plot.face.values,j=vals2plot.j.values,i=vals2plot.i.values)
    return vals2plot,cords2plot,mask
    
## Grafica una variable del modelo LLC4320 para una caja (región), tiempo y profundidad dados
## Ya que estamos usando LLC4320 en su formato "faces", cada cara se grafica una a una en la misma figura, por lo que se deben
## ... obtener los valores mínimo y máximo de la variable a graficar, para que la paleta (colorbar) sea igual para cada cara
## Limitaciones: Por ahora solo grafica usando plt.contourf + las limitaciones de 'readvar_from_latlonbox'
def plotvar_from_latlonbox(min_lon,min_lat,max_lon,max_lat,model_var,time=0,k=0,cmap="autumn_r",n_levels=150):
    vals2plot,cords2plot,_ = readvar_from_latlonbox(min_lon,min_lat,max_lon,max_lat,model_var,time,k)
    
    # Calculamos el mínimo y máximo de la variable en toda la caja, para tener un mismo colorbar
    vmin = vals2plot.min(skipna=True).values
    vmax = vals2plot.max(skipna=True).values
    vcenter = 0 if vmin<0 else (vmin+vmax)/2
    print("Colorbar values (min,center,max):",vmin,vcenter,vmax)
    
    # Configuramos el colorbar
    cmap = plt.cm.get_cmap(cmap)
    divnorm = colors.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    plt.figure(figsize=(20,12))
    dt_str = np.datetime_as_string(vals2plot.time.values, unit='D')
    print("Plotting from {} faces".format(len(vals2plot.face.values)))
    for face in vals2plot.face.values:
        vals = vals2plot.sel(face=face).values
        lon_area = cords2plot.sel(face=face).lon.values
        lat_area = cords2plot.sel(face=face).lat.values
        c_temp = plt.contourf(lon_area,lat_area,vals,levels=n_levels,norm=divnorm,cmap=cmap)
    plt.axis([min_lon, max_lon, min_lat, max_lat])
    plt.colorbar(c_temp)
    plt.title("{} at {}".format(model_var,dt_str))
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


