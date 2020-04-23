from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
mpl.rcParams['font.size'] = 16

def plot_filtered_daily(id,rob_season='daily',t=0,Lt=None,show_boxplot=False,show_histogram=False,hist_max_x=None,hist_bins=20):
    dat = Dataset("/home/antonio/Tesis/Rob_k_filtered/{}/days/Rob_{}.nc".format(id,rob_season),'r')
    
    lat_r = dat["lat"]
    lon_r = dat["lon"]
    f_cor = dat["f_coriolis"]
    RV0_r = dat["RV"][t,0,:,:]/f_cor
    RVH_r = dat["RV"][t,1,:,:]/f_cor
    Rob_r = np.abs(dat["Rob"][t,:,:])
    RV0_lo_r = dat["RV_Lo"][t,0,:,:]/f_cor
    RVH_lo_r = dat["RV_Lo"][t,1,:,:]/f_cor
    Rob_lo_r = np.abs(dat["Rob_Lo"][t,:,:])
    #Rob_lo_r_avg = np.ma.mean(np.ma.array(Rob_lo_r,mask=Rob_lo_r==0),axis=0)
    RV0_hi_r = dat["RV_Hi"][t,0,:,:]/f_cor
    RVH_hi_r = dat["RV_Hi"][t,1,:,:]/f_cor
    Rob_hi_r = np.abs(dat["Rob_Hi"][t,:,:])
    #Rob_hi_r_avg = np.ma.mean(np.ma.array(Rob_hi_r,mask=Rob_hi_r==0),axis=0)
    
    nrows=3
    fig = plt.figure(figsize=(20,4*nrows))

    min_RV0 = min(np.min(RV0_r),np.min(RV0_lo_r),np.min(RV0_hi_r))
    max_RV0 = max(np.max(RV0_r),np.max(RV0_lo_r),np.max(RV0_hi_r))
    min_RVH = min(np.min(RVH_r),np.min(RVH_lo_r),np.min(RVH_hi_r))
    max_RVH = max(np.max(RVH_r),np.max(RVH_lo_r),np.max(RVH_hi_r))
    divnorm_rv = colors.DivergingNorm(vmin=min(min_RV0,min_RVH), vcenter=0, vmax=max(max_RV0,max_RVH))
    #divnorm2 = colors.DivergingNorm(vmin=min_RVH, vcenter=0, vmax=max_RVH)

    min_Rob = min(np.min(Rob_lo_r),np.min(Rob_hi_r),np.min(Rob_r))
    max_Rob = max(np.max(Rob_lo_r),np.max(Rob_hi_r),np.max(Rob_r))
    mean_Rob = (np.mean(Rob_lo_r)+np.mean(Rob_hi_r)+np.mean(Rob_r) )/3
    divnorm_rob = colors.DivergingNorm(vmin=min_Rob,vcenter=(min_Rob+max_Rob)/2,vmax=max_Rob)
    #divnorm_rob = colors.LogNorm(vmin=min_Rob+5e-3,vmax=max_Rob)

    ## Sin Filtrar

    # RV en z=0
    ax1 = fig.add_subplot(nrows,3,1)
    _c1 = plt.pcolormesh(lon_r,lat_r,RV0_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c1,ax=ax1)

    # RV en z=H
    ax2 = fig.add_subplot(nrows,3,2)
    _c2 = plt.pcolormesh(lon_r,lat_r,RVH_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c2,ax=ax2)

    # Ro_b
    ax3 = fig.add_subplot(nrows,3,3)
    #divnorm3 = colors.DivergingNorm(vmin=np.min(Rob_r), vcenter=0, vmax=np.max(Rob_r))
    _c3 = plt.pcolormesh(lon_r,lat_r,Rob_r,norm=divnorm_rob,cmap=plt.cm.Greys) # plt.cm.RdBu
    fig.colorbar(_c3,ax=ax3)

    ## Bajas frecuencias espaciales

    # RV en z=0
    ax1 = fig.add_subplot(nrows,3,4)
    _c1 = plt.pcolormesh(lon_r,lat_r,RV0_lo_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c1,ax=ax1)

    # RV en z=H
    ax2 = fig.add_subplot(nrows,3,5)
    _c2 = plt.pcolormesh(lon_r,lat_r,RVH_lo_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c2,ax=ax2)

    # Ro_b
    ax3 = fig.add_subplot(nrows,3,6)
    _c3 = plt.pcolormesh(lon_r,lat_r,Rob_lo_r,norm=divnorm_rob,cmap=plt.cm.Greys) # plt.cm.RdBu
    fig.colorbar(_c3,ax=ax3)

    ## Altas frecuencias espaciales

    # RV en z=0
    ax1 = fig.add_subplot(nrows,3,7)
    _c1 = plt.pcolormesh(lon_r,lat_r,RV0_hi_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c1,ax=ax1)

    # RV en z=H
    ax2 = fig.add_subplot(nrows,3,8)
    _c2 = plt.pcolormesh(lon_r,lat_r,RVH_hi_r,norm=divnorm_rv,cmap=plt.cm.RdBu_r) # plt.cm.RdBu
    fig.colorbar(_c2,ax=ax2)

    # Ro_b
    ax3 = fig.add_subplot(nrows,3,9)
    _c3 = plt.pcolormesh(lon_r,lat_r,Rob_hi_r,norm=divnorm_rob,cmap=plt.cm.Greys) # plt.cm.RdBu
    fig.colorbar(_c3,ax=ax3)

    fig.show()
    
    ## Plots
    
    if Lt is None:
        lt_label = "L_t".format(Lt)
    else:
        lt_label = "{}~km".format(Lt)
        
    gt_label = r'$> {lt}$'.format(lt=lt_label)
    lt_label = r'$< {lt}$'.format(lt=lt_label)
        
    if show_boxplot:
        # Graficamos la distribución de Ro_b en el área, para escalas cortas y largas -- todos los tiempos disponibles
        plt.figure(figsize=(7,5))
        Rob_lo_vec = np.abs(dat["Rob_Lo"]).flatten()
        Rob_hi_vec = np.abs(dat["Rob_Hi"]).flatten()
        plt.boxplot([Rob_lo_vec,Rob_hi_vec],labels=[gt_label,lt_label],showmeans=False,whis=[5,95],showfliers=False)
        plt.show()
    if show_histogram:
        # Lo mismo de arriba, pero en histogramas
        plt.figure(figsize=(7,5))
        Rob_lo_vec = np.abs(dat["Rob_Lo"]).flatten()
        Rob_lo_vec_ma = np.ma.masked_where(Rob_lo_vec < 0.05 , Rob_lo_vec)
        Rob_hi_vec = np.abs(dat["Rob_Hi"]).flatten()
        Rob_hi_vec_ma = np.ma.masked_where(Rob_hi_vec < 0.05 , Rob_hi_vec)

        plt.hist([Rob_lo_vec_ma,Rob_hi_vec_ma],bins=hist_bins,label=[gt_label,lt_label],
                 density=True,histtype='bar',log=True,alpha=0.75,color=['brown','teal'])
        plt.legend(prop={'size': 15})
        
        if hist_max_x is not None:
            plt.xlim(0,hist_max_x)
        
        plt.show()
        
