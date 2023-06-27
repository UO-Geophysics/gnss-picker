#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:22:22 2022

@author: sydneydybing
"""
from obspy.core import Stream, read, UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs

# Trying this first with just CCCC and July 4 (the M7.1)

path_to_files = '/hdd/Ridgecrest/mseeds/'

stas = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/GNSS_stas.txt', usecols = [2], dtype = str)
print(stas)
chans = ['e', 'n', 'u']
dates = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/eq_dates.txt', dtype = str)

# test_stas = ['CCCC']
# laptop_stas = ['BEPK', 'CCCC', 'P580', 'P594', 'P595']
# test_chans = ['e']
# test_dates = ['20190705']
# laptop_dates = ['20190702', '20190703', '20190704', '20190705', '20190706', '20190707']

for sta in stas:
    
    for chan in chans:
        
        for date in dates:

            try:
                
                print(sta)
                print(chan)
                print(date)
                
                st = read(path_to_files + sta + '/' + sta + '.' + chan + '.' + date + '.mseed')
                # print('read successfully')
                # print(st[0].stats.npts)
                
                st_copy = st.copy()
                st_merge = st_copy.merge(fill_value = 'interpolate')
                print(st_merge[0].stats.npts)
                
                # st.plot(starttime = UTCDateTime('2019-07-05T01:08:00.000000Z'), endtime = UTCDateTime('2019-07-05T01:08:00.000000Z') + 256)
                # st_merge.plot(starttime = UTCDateTime('2019-07-05T01:08:00.000000Z'), endtime = UTCDateTime('2019-07-05T01:08:00.000000Z') + 256)
                
                tr = st_merge.copy()[0]
                # print(tr.stats)
                
                n = -1
                
                for windowed_tr in tr.slide(window_length = 127, step = 127):
                    
                    print("---")  
                    print(windowed_tr)
                    n += 1
                    print(n)
                    
                    data = windowed_tr.data
                    print(data.shape)
                    
                    # Line needs to be changed to make these folders if they don't already exist
                    if path.exists('/hdd/Ridgecrest/eq_days_split/' + sta + '/' + chan + '_split_mseeds/' + date) == False:
                        makedirs('/hdd/Ridgecrest/eq_days_split/' + sta + '/' + chan + '_split_mseeds/' + date)
                    
                    windowed_tr.write('/hdd/Ridgecrest/eq_days_split/' + sta + '/' + chan + '_split_mseeds/' + date + '/' + sta + '.' + chan + '.' + date + '.' + str(n) + '.mseed', format = 'MSEED')
                
            except:
                
                print('Missing station, channel, or date')

    
    
    
# sta_list = []
# chan_list = []
# date_list = []
# starttime_list = []
# endtime_list = []
# n_list = []

# sta_list.append(test_sta)
# chan_list.append(test_chan)
# date_list.append(test_date)
# starttime_list.append(windowed_tr.stats.starttime)
# endtime_list.append(windowed_tr.stats.endtime)
# n_list.append(n)
    
    
    
    