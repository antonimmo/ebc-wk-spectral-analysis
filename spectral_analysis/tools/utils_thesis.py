import re
import numpy as np
from os import listdir

def get_latlonid(fname,log=False):
	lat,lon,s_id = None,None,None
	#LON
	lon_ = re.search('LON_(.+?)_',fname)
	if lon_:
		lon = float(lon_.group(1))
	#LAT
	lat_ = re.search('LAT_(.+?).nc',fname)
	if lat_:
		lat = float(lat_.group(1))
	# ID
	s_id_ = re.search('Spectrum_(.+?)_at',fname)
	if s_id_:
		s_id = s_id_.group(1)

	if log:
		print(fname)
		print("{}/{}/{}".format(lat,lon,s_id))

	return lat,lon,s_id

def igw_bm_partition_k(kh,f,max_freq,Nbv=0.8594,H=4,log=True):
	igw10 = igw_disp_rel(kh,f,10,Nbv=Nbv,H=H,log=log)
	f_abs = np.abs(f)
	return [min(freq,max_freq) for freq in igw10]
	
## Units:
## kh,H: km
## f,Nbv: cph
def igw_disp_rel(kh,f,nn,Nbv=0.8594,H=4,log=True):
	if log:
		print("N={0:.3f},H={1:.3f} - mode {2}".format(Nbv,H,nn))
	#khnp_2 = (kh*H/(nn*np.pi))**2
	kh = (2*np.pi)*kh 	# Transform to rad/km
	khnp_2 = (nn*np.pi/(kh*H))**2
	igw_n = np.sqrt( (Nbv**2 + (f**2)*khnp_2)/(1+khnp_2) )
	#cn = (Nbv*H/(nn*np.pi))
	#igw_n = np.sqrt( (f**2 + (cn*kh)**2))/(1) )
	return igw_n
	#return np.sqrt( (f**2 + (kh*nn)**2) )

# Tested with https://www.mt-oceanography.info/Utilities/coriolis.html
def coriolis(lat):
    omg = (1.)/24 	# cicles/day
    return 2*(omg)*np.sin((lat*3.141519)/180)

def listdir_withext(folder, ext):
	for fname in sorted(listdir(folder)):
		if fname.endswith(ext):
			#print(ext+" "+fname)
			yield fname
