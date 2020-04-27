import json
import numpy as np
from os import listdir
# Imports within the same package
from .common_vars import DATA_FOLDER
from ..tools.utils_thesis import get_latlonid

seasons_ = ['ASO','JFM']
vars_ = ['DIV','KE','RV','SSH']

for season in seasons_:
	for var in vars_:
		folder = "{}/{}".format(season,var)
		print(folder)
		data = {}
		for fname in listdir(folder):
			#print("{}/{}".format(folder,fname))
			if fname.endswith('.nc'):
				lat,lon,s_id = get_latlonid(fname)
				#print(lat)
				lat_f = float(lat)
				season_h = 'winter'
				if (lat_f>=0 and season=='ASO'):
					season_h = 'summer'
				elif (lat_f<0 and season == 'JFM'):
					season_h = 'summer'
				data[s_id] = {
					'fname': fname,
					'lat': lat,
					'lon': lon,
					'var': var,
					'season': season,
					'season_hemisphere': season_h
				}

		with open("{}/spectra/db/all_{}_{}.json".format(DATA_FOLDER,season,var),'w') as out:
			json.dump(data,out)
