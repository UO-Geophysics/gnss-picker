#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:24:43 2021

@author: sydneydybing
"""
import numpy as np
import h5py

long_noise_data = h5py.File('/Users/sydneydybing/GNSSProject/Noise_data/Random_samples/npys_from_tun/all_npys_1.hdf5', 'r')

all_noise_data = long_noise_data['all_noise_run1']
print(len(all_noise_data))

short_noise_data = long_noise_data['all_noise_run1'][2398800:2668800]
print(len(short_noise_data))

h5f = h5py.File('/Users/sydneydybing/NewPGDDistrib/new_sta_grid/270k_noise.hdf5', 'w')  
h5f.create_dataset('270k_noise', data = short_noise_data)
h5f.close()