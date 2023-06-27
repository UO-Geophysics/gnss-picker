#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:36:15 2021

@author: sydneydybing
"""
from obspy import read,Stream,UTCDateTime
import numpy as np
from mudpy.hfsims import windowed_gaussian,apply_spectrum
from mudpy.forward import gnss_psd
import matplotlib.pyplot as plt
import h5py
from glob import glob

rupts = np.genfromtxt('/hdd/rc_fq/ruptures.txt',dtype='str') # VAL
# rupts = np.genfromtxt('/Users/sydneydybing/GNSSProject/reformat_9-8/shorter_rupts_list.txt', dtype='str') # LAPTOP, 5 rupts
test_rupts = ['rupt.000004', 'rupt.000006', 'rupt.000009', 'rupt.000011' ,'rupt.000014']

stas = np.genfromtxt('/hdd/rc_fq/stas_alone.txt', dtype = 'str') # VAL
# stas = np.genfromtxt('/Users/sydneydybing/GNSSProject/reformat_9-8/stations_5cm_PGD.txt', dtype = 'str') # LAPTOP
test_stas = ['P595', 'P594', 'CCCC']
# test_stas = ['P595']
# print(stas)

data_list = []
k = 0
# print(k)

mag_list = []
sta_list = []
rupt_list = []

for krupt in range(len(rupts)):
# for krupt in range(len(test_rupts)):
    
    # rupt = test_rupts[krupt]
    rupt = rupts[krupt]
    
    print('-------------------------------------------------------------')
    print('Rupture = ' + rupt)
    print('-------------------------------------------------------------')
    print(' ')
    
    log = glob('/hdd/rc_fq/new_distrib/output/ruptures/' + rupt + '.log') # TUN
    # log = glob('/Users/sydneydybing/GNSSProject/reformat_9-8/rupts/' + rupt + '.log') # LAPTOP
    
    log = open(log[0],'r')
    line = log.readlines()
    
    mag = str(line[15][21:27])
    # print(mag)
    
    for ista in range(len(stas)):
    # for ista in range(len(test_stas)):
        
        arrivals = np.genfromtxt('/hdd/rc_fq/nd_p_arrivals/allstas_arrival_times_' + rupt + '.csv', dtype='U') # VAL
        # arrivals = np.genfromtxt('/Users/sydneydybing/GNSSProject/reformat_9-8/arrival_times/arrival_times_' + rupt + '.csv', dtype='U') # LAPTOP
        # print(arrivals)
        
        # sta = test_stas[ista]
        sta = stas[ista]
        # print(sta)
        
        # Getting PGD info
        
        summary_file_path = '/hdd/rc_fq/new_distrib/output/waveforms/' + rupt + '/_summary.' + rupt + '.txt'
        # print(summary_file_path)
        
        summary_file_stations = np.genfromtxt(summary_file_path, skip_header=1, dtype = str, usecols = 0)
        
        j = np.where(summary_file_stations == sta)[0]
        
        summary_file = np.genfromtxt(summary_file_path, skip_header=1, dtype = str)
        # print(summary_file)
        pgd = float(summary_file[j][0][6])
        # print(station)
        # print(summary_file[j][0][0])
        # print(pgd)
            
        # Read in data 
        
        stN = read('/hdd/rc_fq/new_distrib/output/waveforms/' + rupt + '/' + sta + '.LYN.sac') # VAL
        stE = read('/hdd/rc_fq/new_distrib/output/waveforms/' + rupt + '/' + sta + '.LYE.sac') # VAL
        stZ = read('/hdd/rc_fq/new_distrib/output/waveforms/' + rupt + '/' + sta + '.LYZ.sac') # VAL
        
        N_data = stN[0].data
        E_data = stE[0].data
        Z_data = stZ[0].data
        
        ### Zero-pad this data ###
        
        N_data_padded = np.pad(N_data, 128, mode = 'constant')
        E_data_padded = np.pad(E_data, 128, mode = 'constant')
        Z_data_padded = np.pad(Z_data, 128, mode = 'constant')
        
        stN_pad = stN.copy()
        stN_pad[0].data = N_data_padded
        
        stE_pad = stE.copy()
        stE_pad[0].data = E_data_padded
        
        stZ_pad = stZ.copy()
        stZ_pad[0].data = Z_data_padded
        
        npts = stN_pad[0].stats.npts

        ### Trim around the arrival time ###
        
        stas_arrival = np.genfromtxt('/hdd/rc_fq/nd_p_arrivals/allstas_arrival_times_' + rupt + '.csv', usecols=[1], dtype='U') # VAL
        # stas_arrival = np.genfromtxt('/Users/sydneydybing/GNSSProject/reformat_9-8/arrival_times/arrival_times_' + rupt + '.csv', usecols=[1], dtype='U') # LAPTOP
 
        i = np.where(stas_arrival == sta)[0]
        arrival = arrivals[i,2][0]

        # Grab the arrival time

        arr_time = UTCDateTime(arrival)
        arr_time = arr_time + 128 # To account for padding at front
        starttime = arr_time - 128
        endtime = arr_time + 127
        
        stN_trim = stN_pad.trim(starttime, endtime)
        stE_trim = stE_pad.trim(starttime, endtime)
        stZ_trim = stZ_pad.trim(starttime, endtime)
        
        stN_trim_data = stN_trim[0].data
        stE_trim_data = stE_trim[0].data
        stZ_trim_data = stZ_trim[0].data
        
        pick_N = stN_trim_data[128]
        pick_E = stE_trim_data[128]   
        pick_Z = stZ_trim_data[128]

        stN_norm = stN_trim_data - pick_N
        stE_norm = stE_trim_data - pick_E
        stZ_norm = stZ_trim_data - pick_Z    
        
        stN_zeroed = stN_norm
        stN_zeroed[0:128] = 0 
        
        stE_zeroed = stE_norm
        stE_zeroed[0:128] = 0 
        
        stZ_zeroed = stZ_norm
        stZ_zeroed[0:128] = 0 
        
        ### Combine N, E, and Z components into one array ###
        
        comb_data = np.append(stN_zeroed, stE_zeroed)
        comb_data = np.append(comb_data, stZ_zeroed) # Order: N, E, Z
        
        k += 1
        print(k)

        ### Adding new data to an array - each row = new station ### 
    
        data_list.append(comb_data) # Add clean data instead
        
        ### Add magnitude to list
        
        rupt_list.append(str(rupt))
        sta_list.append(str(sta))
        mag_list.append(str(mag))
    
data_array = np.array(data_list)
print('Data array shape:')
print(data_array.shape) # Arrivals at samples 65, 193, and 321
# print(data_array[0])

rupt_array = np.array(rupt_list)
# print(rupt_array.shape)

sta_array = np.array(sta_list)
# print(sta_array.shape)

mag_array = np.array(mag_list)
# print(mag_array.shape)

info_array = np.column_stack((rupt_array, sta_array, mag_array))
print('Info array shape:')
print(info_array.shape)
# print(info_array[0])

# np.save('/home/sdybing/rc_fq/100k_w_mag_test.npy', full_array) # TUN w mag
# np.save('/home/sdybing/rc_fq/100k_w_mag_test.npy', data_array) # TUN
# np.save('/Users/sydneydybing/GNSSProject/reformat_9-8/reformat_test.npy', data_array) # LAPTOP

h5f = h5py.File('/hdd/rc_fq/newstas_data.hdf5', 'w') # VAL
# h5f_data = h5py.File('/Users/sydneydybing/GNSSProject/reformat_9-8/test_reformatted_data.hdf5', 'w') # LAPTOP
h5f.create_dataset('newstas_data', data = data_array)
h5f.close()

np.save('/hdd/rc_fq/newstas_info.npy', info_array) # 






