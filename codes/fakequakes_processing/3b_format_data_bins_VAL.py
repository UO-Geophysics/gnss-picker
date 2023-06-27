#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:13:45 2021

@author: sydneydybing
"""
from obspy import read,Stream,UTCDateTime
import numpy as np
from mudpy.hfsims import windowed_gaussian,apply_spectrum
from mudpy.forward import gnss_psd
import matplotlib.pyplot as plt
import h5py
from glob import glob
import random

path_to_files = '/hdd/rc_fq/nd3/'
project = 'nd3'
rupts = np.genfromtxt(path_to_files + 'ruptures.txt', dtype = 'str') # VAL

data_list = []
k = 0 # counter for cycling through processing loop
b = 0 # counter for number of samples with PGD > 1 m

mag_list = []
sta_list = []
rupt_list = []

### Make a list of all of the paths to the summary files ###

sf_paths = []

for rupt in rupts:
    
    sf_path = path_to_files + project + '/output/waveforms/' + rupt + '/_summary.' + rupt + '.txt'
    # print(sf_path)
    
    sf_paths.append(sf_path)
    
# print(sf_paths)
shuffle_sf_paths = random.sample(sf_paths, len(sf_paths))
# print(shuffle_sf_paths)
# print(len(shuffle_sf_paths))

short_sf_paths = shuffle_sf_paths[0:1]

### Make PGD bins ###

lower_bound = np.arange(0,1,0.01)
upper_bound = np.arange(0.01,1.01,0.01)
bin_counter = np.zeros(len(lower_bound))

n_per_bin = 5000

for sf_path in shuffle_sf_paths:
# for sf_path in short_sf_paths:
    
    rupt = sf_path[36:46]
    # print(' ')
    # print('-------------------------------------------------------------')
    # print('Rupture = ' + rupt)
    # print('-------------------------------------------------------------')
    
    log = glob(path_to_files + project + '/output/ruptures/' + rupt + '.log') # VAL
    
    log = open(log[0],'r')
    line = log.readlines()
    
    mag = str(line[15][21:27])
    # print(mag)

    sf = np.genfromtxt(sf_path, skip_header = 1, dtype = str)
    num_rows = np.arange(0,180,1) # number of rows in summary file (= number of stations)
    
    for num_row in num_rows:
        
        # print(num_row)
        
        sta = sf[num_row][0]
        # print(sta)

        pgd = float(sf[num_row][6])
        print('PGD = ' + str(pgd) + ' m')
        
        if pgd >= 1:
            
            arrivals = np.genfromtxt(path_to_files + 'p_arrivals/allstas_arrival_times_' + rupt + '.csv', dtype='U') # VAL
            # print(arrivals)
                
            # Read in data 
            
            stN = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYN.sac') # VAL
            stE = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYE.sac') # VAL
            stZ = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYZ.sac') # VAL
            
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
            
            stas_arrival = np.genfromtxt(path_to_files + 'p_arrivals/allstas_arrival_times_' + rupt + '.csv', usecols=[1], dtype='U') # VAL
     
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
    
            ### Adding new data to an array - each row = new station ### 
        
            data_list.append(comb_data) # Add clean data instead
            
            ### Add magnitude to list
            
            rupt_list.append(str(rupt))
            sta_list.append(str(sta))
            mag_list.append(str(mag))
            
            print('Added data above 1')
            b += 1
            
            k += 1
            print(k)
            print('-------------------------------------------------------------')
        
        elif pgd < 1:
        
            i_lower = np.where(pgd < upper_bound)[0]
            # print(i_lower)
            i_bin = i_lower[0]
            # print(i_bin)
            print('# samples already in bin: ' + str(bin_counter[i_bin]))
            
            if bin_counter[i_bin] < n_per_bin:
                
                arrivals = np.genfromtxt(path_to_files + 'p_arrivals/allstas_arrival_times_' + rupt + '.csv', dtype='U') # VAL
                # print(arrivals)
                    
                # Read in data 
                
                stN = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYN.sac') # VAL
                stE = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYE.sac') # VAL
                stZ = read(path_to_files + project + '/output/waveforms/' + rupt + '/' + sta + '.LYZ.sac') # VAL
                
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
                
                stas_arrival = np.genfromtxt(path_to_files + 'p_arrivals/allstas_arrival_times_' + rupt + '.csv', usecols=[1], dtype='U') # VAL
         
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
        
                ### Adding new data to an array - each row = new station ### 
            
                data_list.append(comb_data) # Add clean data instead
                
                ### Add magnitude to list
                
                rupt_list.append(str(rupt))
                sta_list.append(str(sta))
                mag_list.append(str(mag))
                
                print('Added data to a bin')
                bin_counter[i_bin] += 1
                print(bin_counter)
                
                k += 1
                print(k)
                print('-------------------------------------------------------------')
                
            elif bin_counter[i_bin] >= n_per_bin:
                
                print('Bin is full!')
                print('-------------------------------------------------------------')

print('Numbers of samples in bins:')
print(bin_counter)
print('Number of samples with PGD > 1 m:')
print(b)

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

h5f = h5py.File(path_to_files + 'nd3_data.hdf5', 'w') # VAL
h5f.create_dataset('nd3_data', data = data_array)
h5f.close()

np.save(path_to_files + 'nd3_info.npy', info_array) # VAL
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        