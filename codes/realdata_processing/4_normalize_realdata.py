#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 22:55:12 2022

@author: sydneydybing
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py

real_data = h5py.File('realdata_data.hdf5', 'r')
real_data = real_data['realdata_data'][:,:] # shape: (12240, 384)
real_meta_data = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pos.npy') # shape: (12240, 6)

# print(real_data[0].shape)
# print(real_meta_data)

nlen = 128

comp1 = real_data[:, :nlen]
comp2 = real_data[:, nlen:2*nlen]
comp3 = real_data[:, 2*nlen:]

stack_data = np.zeros((len(real_data), nlen, 3))

for ii, row in enumerate(stack_data):
            
    # Grabbing out the new batch of data using the shifted timespans
    stack_data[ii, :, 0] = comp1[ii, :] # New N component - row in counter, all 128 columns, 0 for first component. Grabs 128s section from comp1
    stack_data[ii, :, 1] = comp2[ii, :]
    stack_data[ii, :, 2] = comp3[ii, :]
    
# print(stack_data.shape)

norm_data = np.zeros((len(stack_data), 128*3))
print(norm_data.shape)

for krow in range(len(stack_data)):
    
    N_data = stack_data[krow, :, 0]
    E_data = stack_data[krow, :, 1]
    Z_data = stack_data[krow, :, 2]
        
    mean_N = np.mean(N_data[0])
    mean_E = np.mean(E_data[0])
    mean_Z = np.mean(Z_data[0])
    
    norm_N_data = N_data - mean_N
    norm_E_data = E_data - mean_E
    norm_Z_data = Z_data - mean_Z
    
    # plt.plot(N_data, label = 'original')
    # plt.plot(norm_N_data, label = 'norm')
    # plt.legend()
    # plt.show()
    
    comb_data = np.append(norm_N_data, norm_E_data)
    comb_data = np.append(comb_data, norm_Z_data)
    
    norm_data[krow,:] = comb_data
    
print(norm_data.shape)
print(norm_data)

h5f = h5py.File('norm_realdata.hdf5', 'w') 
h5f.create_dataset('norm_realdata', data = norm_data)
h5f.close()

    
    
    
        

    
    
    
    
    
