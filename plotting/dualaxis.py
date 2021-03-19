from matplotlib import pyplot as plt
import seaborn as sns

def multi_plot(plots, x, figsize)
  rows = len(plots)
  

  with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(rows,1,figsize=(20,12))
    
    for i,plot in enumerate(plots):
      currAx = ax[i]
      plt.sca(currAx)
      if plot.dual:
        

    
    plt.sca(ax[0])
    ax[0].set(xlim=(tMin,tMax), ylim=(minHf,maxHf), autoscale_on=False)
    line1, = ax[0].plot(time_hr,stats["hflux_mean"],label='avg(oceQnet)')
    plt.axhline(y=0,ls='--',c='k',alpha=0.5)
    addColorBar(ax[0],s)
    ax2 = ax[0].twinx()
    line2, = ax2.plot(time_hr,stats["hbl_mean"],c='r',label="KPPhbl")
    ax2.grid(b=False,axis='both')
    plt.legend((line1,line2),("avg(oceQnet)","KPPhbl"))
    ax[0].set_aspect('auto')
    plt.title('net surface heat flux into the ocean (+=down), >0 increases theta / KPPhbl')
