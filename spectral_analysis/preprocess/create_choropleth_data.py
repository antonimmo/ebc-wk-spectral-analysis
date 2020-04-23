import json
import numpy as np
from os import mkdir
# Imports within the same package
from .common_vars import data_folder
from ..tools.spectral_analysis_tools import open_ds_kwe
from ..tools.utils_thesis import get_latlonid,listdir_withext

def mkdir_(folder):
	print(folder)
	try:
		mkdir(folder)
	except (FileExistsError):
		print('Folder already exists -- Creation skipped')
		pass

def latlon_ft(txt,var,season,a=6,height4lat=None):
	#print(txt)
	lat,lon,s_id = get_latlonid(txt)

	left = lon-a/2
	right = lon+a/2
	if height4lat is None:
		up = lat+a/4
		down = lat-a/4
		height = a/2
	else:
		height = height4lat[lat]
		up = lat+height/2
		down = lat-height/2

	typeStr = 'Polygon'
	coords = [[
				[left,down],
				[left,up],
				[right,up],
				[right,down],
				[left,down]
	]]

	# Return
	return {
		'type': 'Feature',
		'id': int(s_id),
		'properties':{
			'var': var,
			'season': season,
			'lat': lat,
			'lon': lon,
			's_id': int(s_id),
			'height_km': height*20000/180,
			'width_km': (a*40000/360)*np.cos(lat*np.pi/180)
		},
		'geometry':{
			'type': typeStr,
			"coordinates": coords
		}
	}

# Square height depends on latitude
def calculate_square_height(folder):
	lats = list(sorted(set([get_latlonid(fname)[0] for fname in listdir_withext(folder,".nc")])))
	n=len(lats)
	print(n, lats)
	# System to solve: h_{k+1}+h_{k} = P_{k+1}-P_{k} (k=1...n-1); h_n-h_1=0; P=lats,h=H/2
	A = np.zeros([n,n])
	P = np.zeros([n,1])
	for k in range(0,n-1):
		A[k,k+1]=1
		A[k,k]=1
		P[k,0]=lats[k+1]-lats[k]
	A[n-1,n-1]=1
	A[n-1,4]=-1/3
	A[n-1,5]=-1/3
	A[n-1,6]=-1/3
	#print(A)
	#print(P)
	H=np.linalg.solve(A,P)*2
	#print('H=')
	#print(H)
	return {l:h[0] for l,h in zip (lats,H)}

## Calculations to be made on each tile

def avg(fname):
	k,omega,data = open_ds_kwe(fname,False)
	return np.average(data)

def rms(fname):
	k,omega,data = open_ds_kwe(fname,False)
	return np.sqrt(np.mean(np.abs(data)**2))

def stdev(fname):
	k,omega,data = open_ds_kwe(fname,False)
	return np.std(data)

def sum(fname):
	k,omega,data = open_ds_kwe(fname,False)
	return np.sum(data)

# Creates the necessary files
def generate_files(out_folder,season,var,prop):
	folder = season+'/'+var
	print(prop,folder)
	## GEO-DATA (where square tiles are located on map)
	h4lat = calculate_square_height(folder)
	latlon_features = [latlon_ft(fname,var,season,height4lat=h4lat) for fname in listdir_withext(folder,".nc")]
	geo_data = {
		'type': 'FeatureCollection',
		'features': latlon_features,
		'bbox':[-215,-90,145,90]
	}
	with open('{}/{}_{}_geo.json'.format(out_folder,var,season),'w') as out:
		json.dump(geo_data,out)

	## Data for each point (first we use simple calculations in order to test functionality)
	prop = prop.lower()
	if prop not in propList:
		raise Exception('Unimplemented calculation for property: '+prop)
	with open('{}/{}_{}_{}.csv'.format(out_folder,var,season,prop),'w') as out:
		out.write('Id,'+prop)
		for fname in listdir_withext(folder,".nc"):
			lat,lon,s_id = get_latlonid(fname)
			file_ = '{}/{}/{}'.format(season,var,fname)
			print(season,var,prop,s_id)
			if prop == 'avg':
				val = avg(file_)
			elif prop =='rms':
				val = rms(file_)
			elif prop =='stdev': 
				val = stdev(file_)
			elif prop =='sum': 
				val = sum(file_)
			out.write('\n{},{}'.format(s_id,val))

## ** Program 
propList = ['avg','rms','stdev','sum']

out_folder = '{}/map_data'.format(data_folder)
mkdir_(out_folder)	# Creates output folder if it doesn't exist
for season in ['ASO','JFM']:
	for var in ['DIV','KE','RV','SSH']:
		#generate_files(out_folder,season,var,"sum")
		for prop in ['avg','rms','stdev']:
			generate_files(out_folder,season,var,prop)

#season = 'JFM'
#var = 'SSH'
#prop = 'stdev'
#generate_files(season,var,prop)


