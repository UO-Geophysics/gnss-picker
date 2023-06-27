#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:34:41 2022

@author: sydneydybing
"""
import numpy as np
from obspy.core import UTCDateTime

# Associate P and S waves somehow?

event_catalog = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/events.txt', dtype = 'U')
# print(event_catalog)

event_ID_list = []
mag_list = []
stas_list = []
P_arrivals_list = []
S_arrivals_list = []

# a = 0

for kevent in range(len(event_catalog)):

    # print('')
    print('----------------------')
    # print('----------------------')

    event_ID = event_catalog[kevent,0]
    magnitude = event_catalog[kevent,5]
    origin_time = event_catalog[kevent,1]
    origin_time = UTCDateTime(origin_time)
    
    print(event_ID)
    
    GNSS_stas = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_stas.txt', dtype = 'U')
    # print(GNSS_stas)
    
    # P_traveltimes = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_traveltimes/P_waves/' + str(event_ID) + '.npy')
    # print(P_traveltimes)
    
    for ksta in range(len(GNSS_stas)):
        
        sta = GNSS_stas[ksta,2]
        # print('')
        print('----------------------')
        print(sta)
        # print('----------------------')
        stas_list.append(sta)
        
        mag_list.append(magnitude)
        event_ID_list.append(event_ID)
        
        try:
        
            P_traveltimes = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_traveltimes/P_waves/' + str(event_ID) + '.npy')
            P_tt_stas = P_traveltimes[:,1]
            # print(P_traveltimes)
            # print(P_tt_stas)
            
            i = np.where(P_tt_stas == sta)[0]
            
            # print('Origin time: ' + str(origin_time))
            
            P_travel_time = P_traveltimes[i,2][0]
            # print('P-wave travel time: ' + str(P_travel_time))
            
            if P_travel_time == 'nan':
                
                P_arrival_time = 'nan'
                # print('P-wave arrival time: ' + str(P_arrival_time))
                
                P_arrivals_list.append(P_arrival_time)
                print(len(P_arrivals_list))
                # a += 1
                # print('Worked! ' + str(a))
            
            else:
                
                P_arrival_time = origin_time + float(P_travel_time)
                # print('P-wave arrival time: ' + str(P_arrival_time))
                
                P_arrivals_list.append(str(P_arrival_time))
                print(len(P_arrivals_list))
                # a += 1
                # print('Worked! ' + str(a))
                
        except:
            
            P_arrivals_list.append('nan')
            print(len(P_arrivals_list))
            # a += 1
        
        try:
        
            S_traveltimes = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_traveltimes/S_waves/' + str(event_ID) + '.npy')
            S_tt_stas = S_traveltimes[:,1]
            
            k = np.where(S_tt_stas == sta)[0]
            
            S_travel_time = S_traveltimes[k,2][0]
            # print('S-wave travel time: ' + str(S_travel_time))
            
            if S_travel_time == 'nan':
                
                S_arrival_time = 'nan'
                # print('S-wave arrival time: ' + str(S_arrival_time))
                
                S_arrivals_list.append(S_arrival_time)
                print(len(S_arrivals_list))
                # a += 1
                # print('Worked! ' + str(a))
            
            else:
                
                S_arrival_time = origin_time + float(S_travel_time)
                # print('S-wave arrival time: ' + str(S_arrival_time))
                
                S_arrivals_list.append(str(S_arrival_time))
                print(len(S_arrivals_list))
                # a += 1
                # print('Worked! ' + str(a))
                
        except:
                
            S_arrivals_list.append('nan')
            print(len(S_arrivals_list))
            # a += 1

# print(stas_list)
# print(P_arrivals_list)

print(len(event_ID_list))
print(len(mag_list))
print(len(stas_list))
print(len(P_arrivals_list))
print(len(S_arrivals_list))

GNSS_arrival_times = np.column_stack((np.array(event_ID_list), np.array(mag_list), np.array(stas_list), np.array(P_arrivals_list), np.array(S_arrivals_list)))
print(GNSS_arrival_times)

np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_arrival_times.npy', GNSS_arrival_times) # LAP 


# Checking to see if it worked

data = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/GNSS_arrival_times.npy')
print(data)

# i = np.where((data[:,3] != 'error') & (data[:,3] != 'nan'))[0]
i = np.where(data[:,3] != 'nan')[0]
print(i)

print(data[i])
print(len(data[i]))






    
    