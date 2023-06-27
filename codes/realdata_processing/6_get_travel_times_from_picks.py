#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:13:15 2022

@author: sydneydybing
"""
import numpy as np
from obspy import UTCDateTime

event_catalog = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/events.txt', dtype = 'U')
# print(event_catalog)
event_IDs = event_catalog[:,0]
# print(event_IDs)

test_event_IDs = ['37219180', '37219484']



# for event_ID in test_event_IDs:
for event_ID in event_IDs:
    
    P_event_IDs = []
    P_stas_list = []
    P_stalats = []
    P_stalons = []
    P_tts = []
    
    S_event_IDs = []
    S_stas_list = []
    S_stalats = []
    S_stalons = []
    S_tts = []
    
    # if event_ID == '37420709' or event_ID == '37420677' or event_ID == '37420597':
        
    #     pass # no picks
    
    # else:  
        
    print('')
    print('----------------------')
    print('Event:')
    print(event_ID)
    print('----------------------')
    
    try:
    
        P_picks = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/picks_w_sta_locs/P_picks/' + str(event_ID) + '.npy')
        # print(P_picks)
        
        P_stas = P_picks[:,3]
        # print(P_stas)
        
        i = np.where(event_IDs == event_ID)[0]
        # print(i)
        
        # print('Origin time:')
        # origin_time = event_catalog[i,1][0]
        # print(origin_time)
        
        for ksta in range(len(P_stas)):
            
            P_event_IDs.append(event_ID)
            
            P_sta = P_stas[ksta]
            print('Station: ' + str(P_sta))
            P_stas_list.append(P_sta)
            
            latitude = P_picks[ksta,5]
            print(latitude)
            P_stalats.append(latitude)
            
            longitude = P_picks[ksta,6]
            print(longitude)
            P_stalons.append(longitude)
            
            print('Origin time:')
            origin_time = event_catalog[i,1][0]
            print(origin_time)
        
            print('P-wave pick time:')
            pick_time = P_picks[ksta,1]
            print(pick_time)
            
            UTC_ot = UTCDateTime(origin_time)
            UTC_pt = UTCDateTime(pick_time)
            
            # print(UTC_ot)
            # print(UTC_pt)
            
            P_tt = UTC_pt - UTC_ot
            print('P-wave travel time: ' + str(P_tt))
            P_tts.append(P_tt)
            
    except:
        
        print('P-pick error: ' + str(event_ID))
    
    try:
    
        S_picks = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/picks_w_sta_locs/S_picks/' + str(event_ID) + '.npy')
        # print(S_picks)
        
        S_stas = S_picks[:,3]
        # print(S_stas)
        
        i = np.where(event_IDs == event_ID)[0]
        # print(i)
        
        # print('Origin time:')
        origin_time = event_catalog[i,1][0]
        # print(origin_time)
        
        for ksta in range(len(S_stas)):
            
            S_event_IDs.append(event_ID)
            
            S_sta = S_stas[ksta]
            print('Station: ' + str(S_sta))
            S_stas_list.append(S_sta)
            
            latitude = S_picks[ksta,5]
            print(latitude)
            S_stalats.append(latitude)
            
            longitude = S_picks[ksta,6]
            print(longitude)
            S_stalons.append(longitude)
        
            # print('S-wave pick time:')
            pick_time = S_picks[ksta,1]
            # print(pick_time)
            
            UTC_ot = UTCDateTime(origin_time)
            UTC_pt = UTCDateTime(pick_time)
            
            # print(UTC_ot)
            # print(UTC_pt)
            
            S_tt = UTC_pt - UTC_ot
            print('S-wave travel time: ' + str(S_tt))
            S_tts.append(S_tt)
        
        
    except:
        
        print('S-pick error: ' + str(event_ID))

    # print(np.array(P_event_IDs))
    # print(np.array(P_stas_list))
    # print(np.array(P_stalats))
    # print(np.array(P_stalons))
    # print(np.array(P_tts))
    
    print(len(np.array(P_event_IDs)))
    print(len(np.array(P_stas_list)))
    print(len(np.array(P_stalats)))
    print(len(np.array(P_stalons)))
    print(len(np.array(P_tts)))
    
    P_array = np.column_stack((np.array(P_event_IDs), np.array(P_stas_list), np.array(P_stalats), np.array(P_stalons), np.array(P_tts)))
    print(P_array) 
    np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/seismic_traveltimes/P_waves/' + str(event_ID) + '.npy', P_array) # LAP 
        
    print(len(np.array(S_event_IDs)))
    print(len(np.array(S_stas_list)))
    print(len(np.array(S_stalats)))
    print(len(np.array(S_stalons)))
    print(len(np.array(S_tts)))
    
    S_array = np.column_stack((np.array(S_event_IDs), np.array(S_stas_list), np.array(S_stalats), np.array(S_stalons), np.array(S_tts)))
    print(S_array)  
    np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/seismic_traveltimes/S_waves/' + str(event_ID) + '.npy', S_array) # LAP     
    
    
    