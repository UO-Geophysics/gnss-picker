#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:42:23 2021

@author: sydneydybing
"""
import matplotlib.pyplot as plt
from obspy import Stream,read
import UNAVCO_RT
import numpy as np

stas_file = np.genfromtxt('/home/sdybing/GNSS/GNSS_stas.txt', dtype=str)
stas_array = stas_file[:,2]
stas = stas_array.tolist()
test_stas = ['BEPK', 'CCCC']

dates_array = np.genfromtxt('/home/sdybing/GNSS/GNSS_dates.txt', dtype=str)
dates = dates_array.tolist()
test_dates = ['20190606', '20190731']

chans = ['e', 'n', 'u']

path_to_files = '/hdd/Ridgecrest/rt_solutions/'
write_path = '/hdd/Ridgecrest/mseeds/'
figure_path = '/home/sdybing/GNSS/Figures/Parsed_Waveforms/'

for i in range(38,len(dates)):
    
    date = dates[i]
    
    for k in range(0,len(stas)):
        
        sta = stas[k]
        
        try:
        
            df = UNAVCO_RT.parse(path_to_files + sta + '/' + sta + '_rtx_' + date + '.txt')
            
            n,e,u = UNAVCO_RT.unavco2neu(df)
        
            stn,ste,stu = UNAVCO_RT.pandas2obspy(df,n,e,u)
        
            # stn_merge = stn.merge(fill_value=999)
            # ste_merge = ste.merge(fill_value=999)
            # stu_merge = stu.merge(fill_value=999)
            
            # stn_merge = stn.merge(fill_value='interpolate')
            # ste_merge = ste.merge(fill_value='interpolate')
            # stu_merge = stu.merge(fill_value='interpolate')
            
            stn.write(write_path + sta + '/' + sta + '.n.' + date + '.mseed', format='MSEED')
            ste.write(write_path + sta + '/' + sta + '.e.' + date + '.mseed', format='MSEED')
            stu.write(write_path + sta + '/' + sta + '.u.' + date + '.mseed', format='MSEED')
            
            print('Station ' + sta + ', date ' + date + ': SUCCESS')
            
            for chan in chans:
                
                st = Stream()
                st = read(write_path + sta + '/' + sta + '.' + chan + '.' + date + '.mseed')
                
                st_merge = st.merge(fill_value='interpolate')
                
                times = st_merge[0].times()
                data = st_merge[0].data
                
                plt.figure()
                plt.plot(times,data,label=chan)
                plt.title('UNAVCO Real-Time Data, station ' + sta + ', channel ' + chan + ', date ' + date)
                plt.xlabel('Time (s)')
                plt.ylabel('Displacement (m)')
            
                plt.savefig(figure_path + sta + '/' + sta + '.' + chan + '.' + date + '.png', format = 'PNG')
                
        except:
           
            print('Station ' + sta + ', date ' + date + ': ERROR')
    
    