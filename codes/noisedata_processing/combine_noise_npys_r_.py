#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:26:17 2021

@author: sydneydybing
"""
import numpy as np
import h5py

nums = np.arange(0,50,1)
# print(nums)

short_nums = np.arange(0,2,1)

i = 0

array = np.load('/Users/sydneydybing/GNSSProject/Noise_data/Random_samples/npys_from_tun/run2/CPU_' + str(i) + '_noise_samples_2.npy', allow_pickle = True)

for num in nums:
    
    i += 1
    if i == 50:
        break
    
    print(i)
    new_array = np.load('/Users/sydneydybing/GNSSProject/Noise_data/Random_samples/npys_from_tun/run2/CPU_' + str(i) + '_noise_samples_2.npy', allow_pickle = True)
    
    appended = np.r_[array, new_array] # puts rows on top of rows
    # appended = np.append(array, new_array)
    
    array = appended
    print(array.shape)
    # print(array.dtype)
    # print(array[0])

# array.astype(np.float64)
# print(array.dtype)
# print(array[0])    
print(array.shape)
    
h5f = h5py.File('/Users/sydneydybing/GNSSProject/Noise_data/Random_samples/npys_from_tun/all_npys_2.hdf5', 'w')  
h5f.create_dataset('all_noise_run2', data = array)
h5f.close()