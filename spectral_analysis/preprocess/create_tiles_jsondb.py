import json
from os import listdir
# Imports within the same package
from .common_vars import data_folder
from .create_choropleth_data import get_latlonid

jsondb = {}

seasons_ = ['ASO','JFM']
vars_ = ['DIV','KE','RV','SSH']

for season in seasons_:
	for var in vars_:
		folder = "{}/spectra/{}/{}".format(data_folder,season,var):
		for fname in listdir(folder):
			lat,lon,id_ = get_latlonid(fname)
			

