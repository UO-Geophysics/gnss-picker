#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:27:12 2022

@author: sydneydybing
"""
from obspy.core import Stream, read, UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import h5py

# Trying this first with just CCCC and July 4 (the M7.1)

path_to_files = '/hdd/Ridgecrest/eq_days_split/'

stas = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/GNSS_stas.txt', usecols = [2], dtype = str)
chans = ['e', 'n', 'u']
dates = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/eq_dates.txt', dtype = str)
ns = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/ns.txt', dtype = str)

test_stas = ['CCCC']
laptop_stas = ['BEPK', 'CCCC', 'P580', 'P594', 'P595']
test_chans = ['e']
test_dates = ['20190705']
laptop_dates = ['20190702', '20190703', '20190704', '20190705', '20190706', '20190707']

data_list = []
k = -1

sta_list = []
date_list = []
starttime_list = []
endtime_list = []
n_list = []

for sta in stas:
        
    for date in dates:

        for n in ns:
        
            try:
            
                print(sta)
                print(date)
                
                stN = read(path_to_files + sta + '/n_split_mseeds/' + date + '/' + sta + '.n.' + date + '.' + str(n) + '.mseed')
                stE = read(path_to_files + sta + '/e_split_mseeds/' + date + '/' + sta + '.e.' + date + '.' + str(n) + '.mseed')
                stZ = read(path_to_files + sta + '/u_split_mseeds/' + date + '/' + sta + '.u.' + date + '.' + str(n) + '.mseed')
                
                print('read in')
                
                N_data = stN[0].data
                E_data = stE[0].data
                Z_data = stZ[0].data
                
                print('grabbed arrays')
                
                comb_data = np.append(N_data, E_data)
                comb_data = np.append(comb_data, Z_data) # Order: N, E, Z
                
                print('combined data. shape =')
                print(comb_data.shape) 
                
                print('appending to lists')
                data_list.append(comb_data) # Add clean data instead
                
                starttime = stN[0].stats.starttime
                endtime = stN[0].stats.endtime
                
                sta_list.append(sta)
                date_list.append(date)
                starttime_list.append(str(starttime))
                endtime_list.append(str(endtime))
                n_list.append(n)
                
                # Plot
                
                # print(k)
                # if k == 678:
        
                #     plt.figure(figsize=(10,4), dpi=300)
                #     plt.plot(comb_data,label='Real data')
                #     # plt.plot(one_line_noise,label='Noise')
                #     # plt.plot(noisy_data-0.2,label='Noisy data')
                #     # plt.legend(loc = 'upper right')
                    
                #     plt.axvline(127, color = 'black') # last sample of N. E starts at 256
                #     plt.axvline(255, color = 'black') # last sample of E. Z starts at 512
                    
                #     plt.text(25, -0.35, 'N', fontsize = 25, fontweight = 'bold')
                #     plt.text(100, -0.35, 'E', fontsize = 25, fontweight = 'bold')
                #     plt.text(175, -0.35, 'Z', fontsize = 25, fontweight = 'bold')
                    
                #     # plt.show()
                #     plt.savefig('plot.png', format = 'PNG')
                #     plt.close()
                
                # else:
                #     print('pass')
                
                print('counter')
                k += 1
                print(k)
                print('-------------------------------------------------------------')

                
            except:
                
                print('Missing station, channel, or date')

data_array = np.array(data_list)
print('Data array shape:')
print(data_array.shape) # should be something by 384
# print(data_array[0])

sta_array = np.array(sta_list)
# print(sta_array.shape)

date_array = np.array(date_list)
# print(date_array.shape)

starttime_array = np.array(starttime_list)
# print(starttime_array.shape)

endtime_array = np.array(endtime_list)
# print(endtime_array.shape)

n_array = np.array(n_list)
# print(n_array.shape)

info_array = np.column_stack((sta_array, date_array, starttime_array, endtime_array, n_array))
print('Info array shape:')
print(info_array.shape)
# print(info_array[0]) 
# print(info_array[374]) 

h5f = h5py.File(path_to_files + 'realdata_data.hdf5', 'w') 
h5f.create_dataset('realdata_data', data = data_array)
h5f.close()

np.save(path_to_files + 'realdata_info.npy', info_array) 