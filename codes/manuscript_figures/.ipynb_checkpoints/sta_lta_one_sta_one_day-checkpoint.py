#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:39:03 2022

@author: dmelgarm
"""

from obspy.signal.trigger import classic_sta_lta
from obspy import read
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from mudpy.forward import highpass
import matplotlib.dates as mdates
import datetime

dates = ['20190630', '20190701', '20190702', '20190704', '20190705', '20190706', '20190707']
test_dates = ['20190706']

# foreshock_time = datetime.datetime(2019, 7, 4, hour=17, minute=33, second=49, microsecond=0)
# midshock_time = datetime.datetime(2019, 7, 5, hour=11, minute=7, second=53, microsecond=40)
fs_time = datetime.datetime(2019, 7, 6, hour=3, minute=16, second=32, microsecond=480) # 4.97
mainshock_time = datetime.datetime(2019, 7, 6, hour=3, minute=19, second=53, microsecond=40) # 7.1
as1_time = datetime.datetime(2019, 7, 6, hour=3, minute=20, second=41, microsecond=140) # 4.81
as2_time = datetime.datetime(2019, 7, 6, hour=3, minute=23, second=50, microsecond=720) # 4.84
as3_time = datetime.datetime(2019, 7, 6, hour=3, minute=47, second=53, microsecond=420) # 5.5
as4_time = datetime.datetime(2019, 7, 6, hour=3, minute=50, second=59, microsecond=710) # 4.97
as5_time = datetime.datetime(2019, 7, 6, hour=4, minute=13, second=7, microsecond=80) # 4.8
as6_time = datetime.datetime(2019, 7, 6, hour=4, minute=18, second=55, microsecond=790) # 5.44
as7_time = datetime.datetime(2019, 7, 6, hour=4, minute=36, second=55, microsecond=310) # 4.85
as8_time = datetime.datetime(2019, 7, 6, hour=9, minute=28, second=28, microsecond=980) # 4.89

# Keeping aftershocks M4.8 or above and the foreshock on the 6th 

# foreshock_time = mdates.date2num(foreshock_time) 
# midshock_time = mdates.date2num(midshock_time)
fs_time = mdates.date2num(fs_time)
mainshock_time = mdates.date2num(mainshock_time) 
as1_time = mdates.date2num(as1_time)
as2_time = mdates.date2num(as2_time)
as3_time = mdates.date2num(as3_time)
as4_time = mdates.date2num(as4_time)
as5_time = mdates.date2num(as5_time)
as6_time = mdates.date2num(as6_time)
as7_time = mdates.date2num(as7_time)
as8_time = mdates.date2num(as8_time)

threshmax = 10

#kawamoto STA/LTA values
sta = 20
lta = 200

for idx in range(len(test_dates)):
    
    date = test_dates[idx]

    e = read('CCCC.e.' + date + '.mseed')
    n = read('CCCC.n.' + date + '.mseed')
    z = read('CCCC.u.' + date + '.mseed')
    
    # Remove baseline
    e[0].data -= e[0].data[0:60].mean()
    n[0].data -= n[0].data[0:60].mean()
    z[0].data -= z[0].data[0:60].mean()
    
    # Run picker on each channel and each trace
    def picker(data, sta, lta, fc=0.01):
        
        out = np.ones(len(data))*np.nan
        try:
            data = highpass(data, fc, 1.0, 4, zerophase=True)
            out = classic_sta_lta(data, sta, lta)
        except:
            pass
        
        return out
         
    # Pick on each trace individually
    epick = e.copy() 
    npick = n.copy() 
    zpick = z.copy() 
    
    for k in range(len(epick)):
        
        epick[k].data = picker(e[k].data, sta, lta)
        npick[k].data = picker(n[k].data, sta, lta)
        zpick[k].data = picker(z[k].data, sta, lta)
        
    epick.merge(fill_value = np.nan)
    npick.merge(fill_value = np.nan)
    zpick.merge(fill_value = np.nan)
    
    e.merge(fill_value = np.nan)
    n.merge(fill_value = np.nan)
    z.merge(fill_value = np.nan)
    
    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(16,8), dpi = 300)
    # fig = plt.figure(figsize=(16,8))
    
    times = e[0].times(type = 'matplotlib')
    print(times)

    e_data = e[0].data
    n_data = n[0].data
    z_data = z[0].data
    
    # ---------------------------------------------------------------------- #
    
    ax = plt.subplot(6,1,1)
    
    locator = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis = 'y', labelsize = 13)
    
    # plt.gca().xaxis.set_ticklabels([])
    plt.ylabel('Displacement\n(cm)', fontsize = 14)
    plt.title('July 6, 2019 UTC: STA = %ds, LTA = %ds' % (sta,lta), fontsize = 18)
    
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    
    plt.plot(times, e_data*100, c = 'C0', lw=1, label = 'E-W data')
    plt.legend(loc = 'upper right', fontsize = 11)
    
    # ---------------------------------------------------------------------- #
    
    plt.subplot(6,1,2,sharex=ax)

    plt.ylim([0,threshmax])
    plt.tick_params(axis = 'y', labelsize = 13)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axhline(y=5, linestyle='dashed',color='gray',alpha=0.5,linewidth=1.5)
    
    plt.plot(times, epick[0].data,c='black',lw=1, label = 'STA/LTA')
    plt.legend(loc = 'upper right', fontsize = 11)
    
    # ---------------------------------------------------------------------- #
    
    plt.subplot(6,1,3,sharex=ax)
    
    plt.ylabel('Displ. (cm)', fontsize = 14)
    plt.tick_params(axis = 'y', labelsize = 13)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    
    plt.plot(times, n_data*100,c='blueviolet',lw=1, label = 'N-S data')
    plt.legend(loc = 'upper right', fontsize = 11)
    
    # ---------------------------------------------------------------------- #
    
    plt.subplot(6,1,4,sharex=ax)
    
    plt.ylim([0,threshmax])
    plt.tick_params(axis = 'y', labelsize = 13)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axhline(y=5, linestyle='dashed',color='gray',alpha=0.5,linewidth=1.5)
    
    plt.plot(times,npick[0].data,c='black',lw=1, label = 'STA/LTA')
    plt.legend(loc = 'upper right', fontsize = 11)
    
    # ---------------------------------------------------------------------- #
    
    plt.subplot(6,1,5,sharex=ax)
    
    # plt.gca().xaxis.set_ticklabels([])
    plt.ylabel('Displ. (cm)', fontsize = 14)
    plt.tick_params(axis = 'y', labelsize = 13)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    
    plt.plot(times,z_data*100,c='C2',lw=1, label = 'Z data')
    plt.legend(loc = 'upper right', fontsize = 11)
    
    # ---------------------------------------------------------------------- #
    
    plt.subplot(6,1,6,sharex=ax)  
    
    plt.tick_params(axis = 'y', labelsize = 13)
    plt.tick_params(axis = 'x', labelsize = 13)
    
    plt.axvline(x=fs_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=mainshock_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as1_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as2_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as3_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as4_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as5_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as6_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    plt.axvline(x=as7_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    line2 = plt.axhline(y=5, linestyle='dashed',color='gray',alpha=0.5,linewidth=1.5)
    line1 = plt.axvline(x=as8_time,linestyle='dashed',color='red',alpha=0.5,linewidth=1.25)
    
    plt.plot(times,zpick[0].data,c='black',lw=1, label = 'STA/LTA')
    plt.legend(loc = 'upper right', fontsize = 11)
        
    # ---------------------------------------------------------------------- #    
        
    plt.ylim([0,threshmax])
    print(times.shape)
    # ax.set_xlim(times[10000], times[18000]) # Cluster around mainshock
    ax.set_xlim(times[0], times[-1]) # Full
    
    handles, labels = ax.get_legend_handles_labels()
    # print(handles)
    fig.legend([line1,line2], ['Earthquake >M4.8', 'Example threshold = 5'], ncol = 2, loc = 'lower center', fontsize = 14)

    plt.subplots_adjust(hspace=0.1)
    plt.subplots_adjust(bottom = 0.1)

    plt.show()
    # plt.savefig('cluster_w_eqs_0706.png', format = 'PNG')
    # plt.savefig('full_w_eqs_0706.png', format = 'PNG')
    # plt.close()
