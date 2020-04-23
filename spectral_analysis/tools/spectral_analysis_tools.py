import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.axes._subplots import Subplot
from netCDF4 import Dataset
# Imports within the same package
from .utils_thesis import get_latlonid,coriolis,igw_disp_rel,igw_bm_partition_k



#mpl.rcParams['axes.linewidth'] = 2
#mpl.rc('xtick',labelsize=20)
#mpl.rc('ytick',labelsize=20)
#mpl.rc('text',usetex=False)		# No funciona en Colab y tiene poco impacto
#mpl.rcParams['xtick.direction'] = 'out'
#mpl.rcParams['ytick.direction'] = 'out'
#mpl.rcParams['font.family']='DejaVu Sans' # Era Times New Roman, pero ni en Colab ni en OSX funciona

## Frecuencias de marea
#K1 = 1./23.93
M2 = 1./12.4
MK3 = 1./8.17
M4 = 1./6.21
M6 = 1./4.14

# cos(36*pi/180)*(40000/360)*6/2
def open_ds_kwe(fname,log=True):
	dat = Dataset(fname,'r')
	k = dat['Wvnumber'][:]
	omega = dat['Frequency'][:-100]
	E = dat['Spectrum'][:,:-100]
	if log:
		print('**Opening: '+fname)
		print('::::::::: Dimensions :::::::::')
		print('Kiso:',k.shape,k.dtype)
		print('Omega: ',omega.shape,omega.dtype)
		print('E: ',E.shape,E.dtype)
		print('::::::::: Ranges :::::::::')
		print('Kiso range: {0:0.5g},{1:0.5g}'.format(np.min(k),np.max(k)))
		print('Omega range: {0:0.5g},{1:0.5g}'.format(np.min(omega),np.max(omega)))
		print('E range: {0:0.5g},{1:0.5g}'.format(np.min(E),np.max(E)))
		print('::::::::: Ranges (inverse) :::::::::')
		print('1/Kiso range (km): {0:0.5g},{1:0.5g}'.format(1/np.max(k),1/np.min(k)))
		print('1/Omega range (hours): {0:0.5g},{1:0.5g}'.format(1/np.max(omega),1/np.min(omega)))
		print('**\n')
	
	return k,omega,E

	# Nbv = 0.8594 cph ---> 1.5e-3 rad/s Orden "tipico" para la frecuencia de Brunt-Vaisala
	# H: Profundidad media, en km
def plot_wk_integrated(kiso,omega,E,lat,clim,Nbv=0.8594,H=4,wk_only=False,log=True,igw_modes=[1,2,3,4]):
	##::::::::: Setting the figure :::::::::::
	fig = plt.figure(figsize=(7,6))
	if not wk_only:
		ax1 = plt.subplot2grid((3,3),(0,1),colspan=2)
		ax2 = plt.subplot2grid((3,3),(1,0),rowspan=2)
	ax3 = plt.subplot2grid((3,3),(1,1),rowspan=2, colspan=2)

	#::::::::: Frequency-Wavenumber spectrum :::::::
	cs=plt.pcolormesh(kiso,omega,
	              E.T*kiso[None,...]*omega[...,None],
	             cmap='nipy_spectral_r',norm = LogNorm())
	plt.clim(clim)
	ax3.set_yscale('log')
	ax3.set_xscale('log')
	ax3.set_ylim(1/(24*45.),1/3.)
	ax3.set_xlim(1/270.,1/10.)
	ax3.set_yticks([])
	#### Some relevant frequencies
	f = np.abs(coriolis(lat)) ## < ========== Latitude from the name
	ks = np.array([1.e-3,1.])
	# Graficamos la banda f
	ax3.plot(ks,[f,f],'k-',linewidth=2.)
	ax3.text(1/200.,f+0.002,r'$f$',color='k',size='large')

	## Graficamos los modos verticales 1,2,3,4,10 para las ondas internas
	kh = kiso #np.linspace(1/270.,1/5.)
	if igw_modes is None:
		igw_modes = []
	elif type(igw_modes) is not list:
		igw_modes = [igw_modes]
	for mode in igw_modes:
		igw_n = igw_disp_rel(kh,f,mode,Nbv=Nbv,H=H,log=log)
		ax3.plot(kh,igw_n,'k:',linewidth=2.)
	w_partition = igw_bm_partition_k(kh,f,M2,Nbv=Nbv,H=H,log=log)
	ax3.plot(kh,w_partition,'k--',linewidth=2.5)
	
	# Tide constituents
	#for (td,td_label) in zip ([M2,MK3,M4,M6],['M2','MK3','M4','M6']):
	for (td,td_label) in zip ([M2],['M2']):
		ax3.plot(ks,[td, td],'k--',linewidth=2.)
		ax3.text(1/200,td+.0075,td_label,color='k')
	# Trying to set ticks in space domain
	#ax3.set_xlabel('Horizontal Wavenumber k [cpkm]', size=18)  # En numero de onda
	ax3.set_xlabel('Horizontal scales [km]', size='x-large')  # En km
	ax3.set_xticks([1./100.,1./50.,1/25.,1/10.])
	ax3.set_xticklabels(['100','50','25','10'])
	fig.subplots_adjust(right=0.85)
	cbar_ax = fig.add_axes([0.87,.11,0.01, 0.5])
	fig.colorbar(cs, cax=cbar_ax,
	             label=r'$K$ $\times$ $\omega$ $\times$ $\Psi(k,\omega)$ [units$^{2}$/(cpkm $\times$ cph)]')

	if wk_only:
		ax3.set_yticks([1./3., 1./6., 1./12., 1./24., 1./240.])
		ax3.set_yticklabels(['3 h','6 h','12 h','1 d','10 d'])
		ax3.set_ylabel('Time scales',size='x-large')
	else:
		#### :::::: Frequency spectrum :::::::::::
		dk = kiso[1]
		Ef = E[:,:].sum(axis=0)*dk
		ax2.loglog(Ef,omega,linewidth=3.5)
		ax2.set_ylim(1/(24.*45.),1/3.)
		ax2.invert_xaxis()
		# Trying to set ticks in time domain
		#ax2.set_ylabel('Frequency [cph]',size=18)
		ax2.set_yticks([1./4., 1./12., 1./24., 1./240.])
		ax2.set_yticklabels(['4 h','12 h','1 d','10 d'])
		ax2.set_ylabel('Time scales',size='x-large')


		### ::::::: Wavenumber spectrum ::::::::::
		domg = omega[1]
		Ewv = E[:,:].sum(axis=1)*domg
		ax1.loglog(kiso,Ewv,linewidth=3.5)
		ax1.set_xlim(1/270.,1/10.)
		ax1.set_xticklabels(["",""])
		ax1a=ax1.twiny()
		ax1a.set_yscale('log')
		ax1a.set_xscale('log')
		ax1a.set_xlabel("Wavelength [km]",size='x-large')
		ax1a.set_xlim(1/270.,1/10.)
		ax1a.set_xticks([1./100.,1./50.,1/10.])
		ax1a.set_xticklabels(['100','50','10'])

	plt.show()

def calc_bm_igw_k(kh,omega,E,lat,Nbv=0.8594,H=4,log=True):
	f = coriolis(lat)
	f_abs = np.abs(f)

	# IGW mode 10
	#igw_10 = igw_disp_rel(kh,f,10,Nbv=Nbv,H=H,log=log)
	# Partition curve
	w_part = igw_bm_partition_k(kh,f,M2,Nbv=Nbv,H=H,log=log)

	## Hacemos la integral bajo y sobre la curva igw_10 para el modo 10
	# Calculamos deltaOmega
	dw_ = np.diff(omega)
	dw = np.mean(dw_) 
	dw_min = np.min(dw_)
	dw_max = np.max(dw_)
	Ekw = E.T # Transpuesta - me es mas facil pensar que kh va sobre las columnas
	if log:
		print("Calculating IGW(k) and BM(k)")
		print("dw = {0:.6f} (min = {1:.6f}, max = {2:.6f})".format(dw,dw_min,dw_max))
		print('Kh:',kh.shape)
		print('E.T: ',Ekw.shape)
	#	
	mask_igw = np.zeros(Ekw.shape)
	mask_bm = np.zeros(Ekw.shape)
	for kh_i in range(kh.size):
		w_max = w_part[kh_i]
		# Como mask_ ya son float, asignar True equivale a 1, y False a 0
		mask_igw[:,kh_i] = omega>=w_max	# IGW se considera sobre la curva de IGW_10
		mask_bm[:,kh_i] = omega<w_max	# BM se considera sobre la curva de IGW_10
	igw_wk = Ekw*mask_igw
	bm_wk = Ekw*mask_bm

	# Finalmente integramos (sumamos las filas y multiplicamos por el diferencial de omega)
	igw_k = igw_wk[:,:].sum(axis=0)*dw
	bm_k = bm_wk[:,:].sum(axis=0)*dw

	return bm_k,igw_k


def plot_bm_igw_k(kh,omega,E,lat,Nbv=0.8594,H=4,log=True):	
	bm_k,igw_k = calc_bm_igw_k(kh,omega,E,lat,Nbv=Nbv,H=H,log=log)

	if log:
		print("Calculating R(k) = IGW/BM ratio")
	bm_igw = bm_k/igw_k
	
	if log:
		print("Min bm_igw: {}".format(np.min(bm_igw)))
		print("Shapes: igw_wk {}, bm_wk {}".format(igw_wk.shape,bm_wk.shape))
		print("Shapes: igw_k {}, bm_k {}, bm_igw {}".format(igw_k.shape,bm_k.shape,bm_igw.shape))

	## Plot
	plt.semilogx(kh,bm_igw,linewidth=3.5)
	

def get_clim(var):
	if var == 'KE':
	    clim = [1e-6,1e-3]
	elif var == 'SSH':
	    clim = [1e-6,1e-2]
	elif var == 'RV':
	    clim = [1e-12,1e-10]
	elif var == 'DIV':
	    clim = [1e-12,1e-10]

	return clim

def plot_wk_forvar(fname,var,plot_igw_bm=False,Nbv=0.8594,H=4,wk_only=False,log=True):
	#:::::::::::::::: Plot ::::::::::::::::
	clim = get_clim(var)
	kiso,omega,E = open_ds_kwe(fname,log)
	lat,lon,id = get_latlonid(fname)
	if log:
		print("Lat = {}".format(lat))
	plot_wk_integrated(kiso,omega,E,lat,clim,Nbv=Nbv,H=H,wk_only=wk_only,log=log)
	if plot_igw_bm:
		plot_bm_igw_k(kiso,omega,E,lat,Nbv=Nbv,H=H,log=log)

def plot_wk_forseasonvarid(folder,season,var,id,Nbv=0.8594,H=4,wk_only=False,log=True):
	with open("{}/all_{}_{}.json".format(folder,season,var),'r') as f:
		data = json.load(f)
	s_id = str(id)
	#lat = data[s_id]['lat']
	#lon = data[s_id]['lon']
	fname_ = data[s_id]['fname']
	fname = "{}/{}/{}/{}".format(folder,season,var,fname_)
	plot_wk_forvar(fname,var,Nbv=Nbv,H=H,wk_only=wk_only,log=log)


def find_closest_idx(np_arr,val):
	R = np.abs(np_arr-val)
	found = np.where(R==np.amin(R))
	return found[0][0]

def plot_bm_igw_k_forfolderid(folder,season,var,id,scales_km,Nbv=0.8594,H=4,log=True):
	with open("{}/all_{}_{}.json".format(folder,season,var),'r') as f:
		data = json.load(f)
	s_id = str(id)
	lat = data[s_id]['lat']
	lon = data[s_id]['lon']
	fname_ = data[s_id]['fname']
	fname = "{}/{}/{}/{}".format(folder,season,var,fname_)
	k,omega,E = open_ds_kwe(fname,log)
	#if type(scales_km) is tuple:
	#	min_k_idx = find_closest_idx(k,1./scales_km[0])
	#	max_k_idx = find_closest_idx(k,1./scales_km[1])
	#	k = k[min_k_idx:max_k_idx]
	#	E = E[min_k_idx:max_k_idx,:]
	if log:
		print('Kh:',k.shape)
		print('E: ',E.shape)

	if log:
		print("ID = {}, lat = {}, lon = {}, season = {}, var = {}".format(id,lat,lon,season,var))
	plot_bm_igw_k(k,omega,E,lat,Nbv=Nbv,H=H,log=log)
	return k,lat

def plot_bm_igw_k_forseasonvarsid(folder,season,vars,ids,scales_km=(200,10),Nbv=0.8594,H=4,ax=None,show=True,log=True):
	if ax is None:
		ax = plt
	legend_ = vars

	if type(vars) is str:
		if type(ids) is str:	
			kh,_ = plot_bm_igw_k_forfolderid(folder,season,vars,ids,scales_km=scales_km,Nbv=Nbv,H=H,log=log)
			title = "{} for id = {}, season: {}".format(vars,ids,season)
			legend = [vars]
			#ax.legend([vars])
		elif type(ids) is list:
			legend_ = []
			for id_,N,H_ in zip(ids,Nbv,H):
				kh,lat = plot_bm_igw_k_forfolderid(folder,season,vars,id_,scales_km=scales_km,Nbv=N,H=H_,log=log)
				legend_.append("{} ({})".format(id_,lat))
			title = "{}, season: {}".format(vars,season)
			#ax.legend(legends)
	elif type(vars) is list:
		for var in vars:
			kh,_ = plot_bm_igw_k_forfolderid(folder,season,var,ids,scales_km=scales_km,Nbv=Nbv,H=H,log=log)
		title = "id = {}, season: {}".format(ids,season)
		#ax.legend(vars)
	
	ax.semilogx([kh[0],kh[-1]],[1,1],'k--')
	if type(ax) is Subplot:
		ax.set_title(title,size='xx-large')
		scales = [round(1/kh[0],1),200,150,100,75,50,35,25,15,10,5,round(1/kh[-1],1)]
		scales = [scale for scale in scales if (scale<=scales_km[0] and scale>=scales_km[-1])]
		ax.set_xticks(np.divide(1,scales))
		ax.set_xticklabels(["{}".format(int(scale)) if scale%1==0 else "{0:.1f}".format(scale) for scale in scales],size='medium')
		ax.set_xlim(1/scales[0],1/scales[-1])
		#ax.set_ylim([0,2])
		ax.set_xlabel("Wavelength [km]",size='x-large')
	else:
		ax.title(title,size='xx-large')

	ax.legend(legend_)
	if show and (type(ax) is not Subplot):
		plt.show()
