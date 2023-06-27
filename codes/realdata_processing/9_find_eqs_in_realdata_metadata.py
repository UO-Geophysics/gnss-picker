#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:57:32 2022

@author: sydneydybing
"""
import h5py
import numpy as np
from obspy.core import UTCDateTime

real_meta_data = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/realdata_info.npy')

#print(real_meta_data)
# print(len(real_meta_data))

# print(real_data.shape[0])
# print(real_meta_data)   # Need to associate each row in this with a 1 or a 0 if
                        # an earthquake is supposed to be found there
                        
test_meta_data = real_meta_data[1489:4102,:] # BEPK 7/04 17 hrs to 7/07 6 hrs. Indices 1100 to 2900
# print(test_meta_data)

# print(real_meta_data.shape)

event_catalog = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/events.txt', dtype = 'U')
# print(event_catalog)

arrivaltimes = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_arrival_times.npy')

# print(arrivaltimes)
# print(len(arrivaltimes))
# print('-----------------------------')

# i = np.where(arrivaltimes[:,2] == 'BEPK')[0]
# print(i)

# for ki in i:
    
#     print(arrivaltimes[ki,:])

# target_list = []
# target_wavetype_list = []

rows_with_eqs = []
rwe_mags = []
rows_without_eqs = []
p_count_info = []
gauss_positions = []

counter = 0

# for krow in range(len(real_meta_data)):
for krow in range(len(test_meta_data)):
    
    # print('-----------------------------')
    
    # sta = real_meta_data[krow,0]
    # date = real_meta_data[krow,1]
    # starttime = UTCDateTime(real_meta_data[krow,2])
    # # endtime = UTCDateTime(real_meta_data[krow,3])
    
    sta = test_meta_data[krow,0]
    date = test_meta_data[krow,1]
    starttime = UTCDateTime(test_meta_data[krow,2])
    endtime = UTCDateTime(test_meta_data[krow,3])
    
    # print(sta)
    # print(date)
    # print(starttime)
    # print(endtime)
    
    # arrival_stas = arrivaltimes[:,2]
    # p_arrivals = arrivaltimes[:,3]
    # s_arrivals = arrivaltimes[:,4]
    
    arrival_stas = arrivaltimes[:,2]
    # print(arrival_stas)
        
    i = np.where(arrival_stas == sta)[0]
    # print(i)
    
    p_pick_count = 0
    
    for index in i:
        
        # print('-----------------------------')
        
        mag = arrivaltimes[index,1]
        arrival_sta = arrivaltimes[index,2]
        p_arrival = arrivaltimes[index,3]
        s_arrival = arrivaltimes[index,4]
        
        # print(mag)
        # print(arrival_sta)
        # print(p_arrival)
        # print(s_arrival)
        
        if p_pick_count == 0: # If we haven't found any P waves yet in the metadata row:
        
            if p_arrival != 'nan': # If the P wave arrival time in the arrivaltimes array is not 'nan' (aka it exists):
                
                # print('-----------------------------')
                # print('Good P-wave')
                
                p_arrival = UTCDateTime(p_arrival) # Convert the P arrival time from the array to a UTCDateTime object
                # print('P-arrival: ' + str(p_arrival))
                # print('Start time: ' + str(starttime))
                # print('End time: ' + str(endtime))
                
                p_start_delta = p_arrival - starttime # If the P arrival minus the metadata row start time is greater than or equal to 0 but less than 128, the p_arrival is in the sample
                # print(p_start_delta)
                
                if p_start_delta >= 0 and p_start_delta < 128: # If the delta is 0, the Gaussian should be positioned at the first sample (index 0).
                    
                    gauss_position = round(p_start_delta) # Rounded to the nearest whole index
                    p_pick_count += 1 # Since we found an arrival in this metadata row, add one to the pick count
                    
                    print('Index ' + str(index) + ': P in this sample!')
                    # print(krow)
                    # print(mag)
                    rwe_mags.append(mag)
                    # print(index) # this is the index in arrival_times where the eq is
                
                else: # If the delta is greater than 128, the arrival time is not in the metadata array. Skip it.
                    
                    pass
            
            elif p_arrival == 'nan': # If there isn't a calculated arrival time for this row in the arrivaltimes array, skip it.
                
                gauss_position = 'nan' # If there is no p arrival in the sample, the gaussian will not exist so the position is 'nan'.
                pass
        
        elif p_pick_count > 0: # If the count is already greater than 1 because we've found a p wave in this row of the metadata array, we'll skip any extras for making gaussian positions (but still add them to the count so we know they exist).
            
            if p_arrival != 'nan': 
                p_arrival = UTCDateTime(p_arrival) 
                p_start_delta = p_arrival - starttime 

                if p_start_delta >= 0 and p_start_delta < 128:  
                    p_pick_count += 1
                
                else:
                    pass

            elif p_arrival == 'nan': 
                pass
    
    if p_pick_count == 0:
        
        rows_without_eqs.append(krow) # Add the index of the row in the metadata array that doesn't have an earthquake to the list of indices that don't have earthquakes.
        p_count_info.append(p_pick_count) # Add the number of p picks in the metadata array row (zero) to the p_count list.
        gauss_positions.append(gauss_position) # Add the gauss position (in this case 'nan' because there are no picks in this row of the metadata array) to the list of positions.
        
        # These might be VERY WRONG - need to be inside the index loop to not just have only the last value!
        
    elif p_pick_count != 0:
        
        rows_with_eqs.append(krow) # Add the index of the row in the metadata array that does have an earthquake to the list of indices that do have earthquakes.
        p_count_info.append(p_pick_count) # Add the number of p picks in the metadata array row to the p_count list.
        gauss_positions.append(gauss_position) # Add the index of the sample that should have the gaussian peak to the list of gaussian positions.

    counter += 1
    # print(counter)

## Checks to make sure things add up correctly

# print(gauss_positions)
print(rwe_mags)

# print(len(real_meta_data))
# print(len(rows_without_eqs) + len(rows_with_eqs))
# print(len(p_count_info))
# print(len(gauss_positions)) # All four of these should match

# print(len(rows_with_eqs))
# a = np.where(np.array(gauss_positions) != 'nan')[0]
# print(len(a)) # This should match len(rows_with_eqs)

# ### Adding gauss position column to metadata array

# gauss_pos_vector = np.array(gauss_positions).reshape(len(gauss_positions),1)

# new_meta_array = np.append(real_meta_data, gauss_pos_vector, axis = 1)
# print(real_meta_data.shape)
# print(new_meta_array.shape)
# print(new_meta_array)

# # Checking rows that should have the gaussian position added

# print(new_meta_array[rows_with_eqs])
# print(len(new_meta_array[rows_with_eqs]))

# # Checking rows with eqs and mags are same length

# print('November 2022 checks')
# print(len(rows_with_eqs))
# print(len(rwe_mags))
# print(rwe_mags)

# np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pos.npy', new_meta_array)
# np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rowsweqs.npy', np.array(rows_with_eqs))
# np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rwe_mags.npy', np.array(rwe_mags))
    

        
        
        
        
        
        
    

        
        
        
        
    
    


