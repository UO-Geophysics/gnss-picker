#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 01:24:39 2022

@author: sydneydybing
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

'''
Files required to run this code:
    1. Real data (h5py)
    2. Real metadata (npy) from #8 in real data code workflow
    3. Rows with earthquakes (npy) from #8 in real data code workflow
    4. Rows that are the earthquakes the CNN correctly found (txt) (handmade for now)
'''

### Loading the waveforms (real_data) and the associated metadata (real_meta_data)

real_data = h5py.File('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/realdata_data.hdf5', 'r')
real_data = real_data['realdata_data'][:,:] # shape: (12240, 384)
real_meta_data = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pos.npy') # shape: (12240, 6)
# print(real_meta_data[0]) # Columns: station, date, start time, end time, counter, gauss position

### Pieces of the data generator just to get everything formatted to calculate the PGDs and SNRs

nlen = 128 # Window length of each component (since all 3 are stacked side by side in real_data)

comp1 = real_data[:, :nlen] # N-S
comp2 = real_data[:, nlen:2*nlen] # E-W
comp3 = real_data[:, 2*nlen:] # Z

## Stacking the data back together. Don't really remember why I did this but that's how it's set up now lol

stack_data = np.zeros((len(real_data), nlen, 3))

for ii, row in enumerate(stack_data):
            
    stack_data[ii, :, 0] = comp1[ii, :] # New N component - row in counter, all 128 columns, 0 for first component. Grabs 128s section from comp1
    stack_data[ii, :, 1] = comp2[ii, :]
    stack_data[ii, :, 2] = comp3[ii, :]

### Establishing lists to add to as we calculate things

pgds = []
SNRs_N = []
SNRs_E = []
SNRs_Z = []

### Looping through all of the rows in the data to calculate PGD and SNR

for krow in range(len(stack_data)):
    
    N_data = stack_data[krow, :, 0]
    E_data = stack_data[krow, :, 1]
    Z_data = stack_data[krow, :, 2]
    
    p_arrival_index = real_meta_data[krow, 5] # The index in the sample that the P-wave arrives at
    
    if p_arrival_index != 'nan': # This means there was an earthquake in that sample. Non-earthquakes have a P-arrival index of 'nan'
        
        p_arrival_index = int(p_arrival_index)
        
        '''
        In this section, I calculate the mean of the 10 seconds of data (or less)
        before the P-wave arrival, then subtract this from the data, to normalize
        everything before calculating the PGD.
        '''
        
        preeq_mean_end = p_arrival_index # The end of the section before the earthquake to calculate the mean is the P-wave arrival time

        if preeq_mean_end == 0: # If the P-wave arrives right at the start of the sample...
            
            preeq_mean_N = N_data[0] # ...I'm just going to subtract the first value of the sample
            preeq_mean_E = E_data[0]
            preeq_mean_Z = Z_data[0]
        
        elif preeq_mean_end <= 10:
            
            preeq_mean_start = 0 # To avoid negative index start times
            
            preeq_mean_N = np.mean(N_data[preeq_mean_start:preeq_mean_end])
            preeq_mean_E = np.mean(E_data[preeq_mean_start:preeq_mean_end])
            preeq_mean_Z = np.mean(Z_data[preeq_mean_start:preeq_mean_end])
            
        else:
            
            preeq_mean_start = preeq_mean_end - 10
        
            preeq_mean_N = np.mean(N_data[preeq_mean_start:preeq_mean_end])
            preeq_mean_E = np.mean(E_data[preeq_mean_start:preeq_mean_end])
            preeq_mean_Z = np.mean(Z_data[preeq_mean_start:preeq_mean_end])
        
        norm_N_data = N_data - preeq_mean_N
        norm_E_data = E_data - preeq_mean_E
        norm_Z_data = Z_data - preeq_mean_Z
        
        # plt.plot(N_data, label = 'original')
        # plt.plot(norm_N_data, label = 'norm')
        # plt.legend()
        # plt.axvline(x = p_arrival_index, color = 'gray', linestyle = 'dashed')
        # plt.show()
        
        '''
        Calculating the PGD and adding it to the list.
        '''

        pgd = max(np.sqrt(norm_N_data**2 + norm_E_data**2 + norm_Z_data**2))
    
        pgds.append(pgd)
        
        '''
        In this section, I calculate the signal-to-noise ratio of the data. I 
        aim to use a window of 20 seconds before the P-wave arrival time as the 
        noise, and a window of 20 seconds after the P-wave arrival time as the 
        signal. I take the standard deviation of these segments and divide 
        signal/noise (or after/before) to get the SNR.
        
        Sometimes the P-wave arrival time is too close to the start or end of the
        sample, and this causes issues. I've added conditions for these cases.
        '''
        
        preeq_std_end = p_arrival_index # The end of the 20 second 'noise' section before the earthquake is the P-wave arrival index
        
        if preeq_std_end <= 10: # Ask Diego if this is reasonable # Try 10
        
            # If P-wave pick is at zero - can't calculate a pre-eq standard deviation. 
            # OR the P-wave pick is too close to zero, it throws off the SNR values by a LOT.
            
            SNR_N = 'nan' # Just skip it (at least 10 cases for Z component with weird SNRs - one over 10,000!)
            SNR_E = 'nan'
            SNR_Z = 'nan'
        
        elif preeq_std_end > 10 and preeq_std_end <= 20: # If the pre-earthquake noise window is smaller than 20 seconds...
            
            preeq_std_start = 0
            
            posteq_std_start = p_arrival_index # Start the section for the "signal" at the P-wave arrival index
            posteq_std_end = posteq_std_start + 20
            # posteq_std_end = posteq_std_start + p_arrival_index # If the window before is less than 20 because the arrival time is less than 20, this makes the window after that same length
            
            std_before_N = np.std(norm_N_data[preeq_std_start:preeq_std_end]) # Take the standard deviation of the sections for each component
            std_after_N = np.std(norm_N_data[posteq_std_start:posteq_std_end])
            std_before_E = np.std(norm_E_data[preeq_std_start:preeq_std_end])
            std_after_E = np.std(norm_E_data[posteq_std_start:posteq_std_end])
            std_before_Z = np.std(norm_Z_data[preeq_std_start:preeq_std_end])
            std_after_Z = np.std(norm_Z_data[posteq_std_start:posteq_std_end])
            
            if std_before_N == 0 or std_before_E == 0 or std_before_Z == 0: # If any of the denominators are zeros, we get 'inf' in the results
                
                SNR_N = 'nan' # Skip 'em
                SNR_E = 'nan'
                SNR_Z = 'nan'
                
            else: # Calculate the SNR
                
                SNR_N = std_after_N / std_before_N
                SNR_E = std_after_E / std_before_E
                SNR_Z = std_after_Z / std_before_Z
        
        elif preeq_std_end > 20 and preeq_std_end <= 108: # Standard case where the P-wave arrival is nicely in the middle somewhere
            
            preeq_std_start = preeq_std_end - 20
            
            posteq_std_start = p_arrival_index
            posteq_std_end = posteq_std_start + 20
        
            std_before_N = np.std(norm_N_data[preeq_std_start:preeq_std_end])
            std_after_N = np.std(norm_N_data[posteq_std_start:posteq_std_end])
            std_before_E = np.std(norm_E_data[preeq_std_start:preeq_std_end])
            std_after_E = np.std(norm_E_data[posteq_std_start:posteq_std_end])
            std_before_Z = np.std(norm_Z_data[preeq_std_start:preeq_std_end])
            std_after_Z = np.std(norm_Z_data[posteq_std_start:posteq_std_end])
            
            if std_before_N == 0 or std_before_E == 0 or std_before_Z == 0:
                
                SNR_N = 'nan'
                SNR_E = 'nan'
                SNR_Z = 'nan'
                
            else:
                
                SNR_N = std_after_N / std_before_N
                SNR_E = std_after_E / std_before_E
                SNR_Z = std_after_Z / std_before_Z
            
        elif preeq_std_end > 108 and preeq_std_end < 128: # End edge case - the "signal" period is less than 20 seconds long
            
            preeq_std_start = preeq_std_end - 20
            
            posteq_std_start = p_arrival_index
            posteq_std_end = posteq_std_start + (128 - p_arrival_index) # Make the signal period end at the end of the sample at 128 to avoid errors
        
            std_before_N = np.std(norm_N_data[preeq_std_start:preeq_std_end])
            std_after_N = np.std(norm_N_data[posteq_std_start:posteq_std_end])
            std_before_E = np.std(norm_E_data[preeq_std_start:preeq_std_end])
            std_after_E = np.std(norm_E_data[posteq_std_start:posteq_std_end])
            std_before_Z = np.std(norm_Z_data[preeq_std_start:preeq_std_end])
            std_after_Z = np.std(norm_Z_data[posteq_std_start:posteq_std_end])
            
            if std_before_N == 0 or std_before_E == 0 or std_before_Z == 0:
                
                SNR_N = 'nan'
                SNR_E = 'nan'
                SNR_Z = 'nan'
                
            else:
                
                SNR_N = std_after_N / std_before_N
                SNR_E = std_after_E / std_before_E
                SNR_Z = std_after_Z / std_before_Z
            
        else: # Covers if the pick is exactly at 128, the end of the sample.
            
            # Can't get a post-eq std because the earthquake arrives at the end of the sample
            
            SNR_N = 'nan' # Skip 'em (5 cases)
            SNR_E = 'nan'
            SNR_Z = 'nan'
            
        '''
        Add the calculated SNRs (or 'nan's for issues) to the lists.
        '''
            
        SNRs_N.append(SNR_N)
        SNRs_E.append(SNR_E)
        SNRs_Z.append(SNR_Z)
        
    elif p_arrival_index == 'nan': # When there isn't an earthquake in the sample, the P-arrival is 'nan'. Skip 'em
        
        pgds.append('nan')
        SNRs_N.append('nan')
        SNRs_E.append('nan')
        SNRs_Z.append('nan')
        
'''
This next section deals with calculating the averages for the entire set of 
earthquakes and the set of earthquakes which the CNN got correct in testing.
'''

pgds_vector = np.array(pgds).reshape(len(pgds),1) # Turn the list of PGDs we made into a vector...

new_meta_array = np.append(real_meta_data, pgds_vector, axis = 1) # ...And add it to the metadata array

# np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/RealData/Pick_stuff/real_metadata_w_gauss_pos_and_pgds.npy', new_meta_array)

# print(new_meta_array[0]) # Columns: station, date, start time, end time, counter, gauss position, pgd

'''
From code 8 in the real data workflow, we have a NumPy array that just lists the
indices in the entire real_data or real_meta_data array that DO contain earthquakes.
I also handmade a text file containing the indices of the earthquakes that the
CNN actually did find in testing by going through all of the figures. I'm sure 
there's a way to automate this with the training code but I haven't done it yet.
'''

rows_w_eqs = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rowsweqs.npy') # Rows that have earthquakes
correct_eq_inds = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/newtrain_march/more_realdata_norm_testing/correct_indices.txt', dtype = 'int') # Rows that the CNN found earthquakes in

'''
Calculating average PGDs for the groups.
'''

all_eq_pgds = np.asfarray(new_meta_array[rows_w_eqs, 6]) # The PGDs of all of the earthquakes
correct_eq_pgds = np.asfarray(new_meta_array[correct_eq_inds, 6]) # The PGDs of all the earthquakes the CNN found

all_eq_avg_PGD = np.mean(all_eq_pgds)
correct_eq_avg_PGD = np.mean(correct_eq_pgds)

print('Average PGD of all earthquakes: ' + str(round((all_eq_avg_PGD * 100),2)) + ' cm')
print('Average PGD of earthquakes the CNN correctly found: ' + str(round((correct_eq_avg_PGD * 100),2)) + ' cm')
print('-------------------------------------------------------------------')

'''
Calculating average SNRs for all three components of both groups.
'''

SNR_N_vector = np.array(SNRs_N).reshape(len(SNRs_N),1) # Turns the lists of SNRs into vectors...
SNR_E_vector = np.array(SNRs_E).reshape(len(SNRs_E),1)
SNR_Z_vector = np.array(SNRs_Z).reshape(len(SNRs_Z),1)

new_meta_array_a = np.append(new_meta_array, SNR_N_vector, axis = 1) # ...And adds these to make another new metadata array
new_meta_array_b = np.append(new_meta_array_a, SNR_E_vector, axis = 1)
new_meta_array_2 = np.append(new_meta_array_b, SNR_Z_vector, axis = 1)

# print(new_meta_array_2[0]) # Columns: station, date, start time, end time, counter, gauss position, pgd, SNR N component, SNR E, SNR Z

np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pgd_snr.npy', new_meta_array_2)

'''
Because of the edge cases in the loop above, there are some nans in our SNR vectors.
We can't calculate averages with nans, so we need to find the rows with nans and
just remove them for the sake of this calculation.
'''

h = np.where(new_meta_array_2[rows_w_eqs,7] == 'nan') # Finds nans for all earthquakes
non_nan_rows_w_eqs = np.delete(rows_w_eqs, h) # Removes those rows

j = np.where(new_meta_array_2[correct_eq_inds,7] == 'nan') # Finds nans for the earthquakes the CNN found
non_nan_correct_eq_inds = np.delete(correct_eq_inds, j) # Removes those rows

'''
Now I just grab the good SNRs out of the new metadata array and calculate the averages.
'''

all_eq_SNR_N = np.asfarray(new_meta_array_2[non_nan_rows_w_eqs, 7])
all_eq_SNR_E = np.asfarray(new_meta_array_2[non_nan_rows_w_eqs, 8])
all_eq_SNR_Z = np.asfarray(new_meta_array_2[non_nan_rows_w_eqs, 9])

correct_eq_SNR_N = np.asfarray(new_meta_array_2[non_nan_correct_eq_inds, 7])
correct_eq_SNR_E = np.asfarray(new_meta_array_2[non_nan_correct_eq_inds, 8])
correct_eq_SNR_Z = np.asfarray(new_meta_array_2[non_nan_correct_eq_inds, 9])

all_eq_SNR_N_avg = np.mean(all_eq_SNR_N)
all_eq_SNR_E_avg = np.mean(all_eq_SNR_E)
all_eq_SNR_Z_avg = np.mean(all_eq_SNR_Z)

correct_eq_SNR_N_avg = np.mean(correct_eq_SNR_N)
correct_eq_SNR_E_avg = np.mean(correct_eq_SNR_E)
correct_eq_SNR_Z_avg = np.mean(correct_eq_SNR_Z)

print(len(all_eq_SNR_N))
print(len(correct_eq_SNR_N))

print('Average N-S component SNR of all earthquakes: ' + str(round(all_eq_SNR_N_avg,2)))
print('Average N-S component SNR of earthquakes the CNN correctly found: ' + str(round(correct_eq_SNR_N_avg,2)))
print('-------------------------------------------------------------------')

print('Average E-W component SNR of all earthquakes: ' + str(round(all_eq_SNR_E_avg,2)))
print('Average E-W component SNR of earthquakes the CNN correctly found: ' + str(round(correct_eq_SNR_E_avg,2)))
print('-------------------------------------------------------------------')

print('Average Z component SNR of all earthquakes: ' + str(round(all_eq_SNR_Z_avg,2)))
print('Average Z component SNR of earthquakes the CNN correctly found: ' + str(round(correct_eq_SNR_Z_avg,2)))




