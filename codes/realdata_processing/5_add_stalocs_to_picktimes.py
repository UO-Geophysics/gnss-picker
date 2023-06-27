#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:09:44 2022

@author: sydneydybing
"""

from os import path,makedirs
from obspy.clients.fdsn import Client
import obspy.io.quakeml.core as quakeml
from obspy.core.event import read_events
from obspy import UTCDateTime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

event_catalog = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/events.txt', dtype = 'U')
# print(event_catalog)
event_IDs = event_catalog[:,0]
print(event_IDs)

starttime = UTCDateTime("2019-06-01T00:00:00")
endtime = UTCDateTime("2020-07-30T23:59:59")

client = Client("SCEDC")
inventory = client.get_stations(network = "CI,PB", station = "*", channel = '*', starttime = starttime,
                                endtime = endtime, latitude = 35.7695, longitude = -117.5993333, maxradius = 10)

CI_sta_len = np.arange(434)

CI_sta_list = []
CI_sta_lat = []
CI_sta_lon = []
CI_sta_elev = []

for ksta in CI_sta_len:
    
    sta_code = inventory[0][ksta].code
    sta_lat = inventory[0][ksta].latitude
    sta_lon = inventory[0][ksta].longitude
    sta_elev = inventory[0][ksta].elevation
    
    CI_sta_list.append(sta_code)
    CI_sta_lat.append(sta_lat)
    CI_sta_lon.append(sta_lon)
    CI_sta_elev.append(sta_elev)

CI_sta_list_array = np.array(CI_sta_list)
CI_sta_lat_array = np.array(CI_sta_lat)
CI_sta_lon_array = np.array(CI_sta_lon)
CI_sta_elev_array = np.array(CI_sta_elev)

CI_info_array = np.column_stack((CI_sta_list_array, CI_sta_lat_array, CI_sta_lon_array, CI_sta_elev_array))
# print(CI_info_array)

PB_sta_len = np.arange(53)

PB_sta_list = []
PB_sta_lat = []
PB_sta_lon = []
PB_sta_elev = []

for ksta in PB_sta_len:
    
    sta_code = inventory[1][ksta].code
    sta_lat = inventory[1][ksta].latitude
    sta_lon = inventory[1][ksta].longitude
    sta_elev = inventory[1][ksta].elevation
    
    PB_sta_list.append(sta_code)
    PB_sta_lat.append(sta_lat)
    PB_sta_lon.append(sta_lon)
    PB_sta_elev.append(sta_elev)

PB_sta_list_array = np.array(PB_sta_list)
# print(PB_sta_list_array)
PB_sta_lat_array = np.array(PB_sta_lat)
PB_sta_lon_array = np.array(PB_sta_lon)
PB_sta_elev_array = np.array(PB_sta_elev)

PB_info_array = np.column_stack((PB_sta_list_array, PB_sta_lat_array, PB_sta_lon_array, PB_sta_elev_array))
# print(PB_info_array)

test_event_IDs = ['38472279', '37219156']

z = 0
p = 0 
s = 0

for event_ID in event_IDs:
    
    # print(event_ID)
    
    P_eventID_list = []
    P_picktime_list = []
    P_net_list = []
    P_sta_list = []
    P_chan_list = []
    P_stalat_list = []
    P_stalon_list = []
    P_staelev_list = []
    
    S_eventID_list = []
    S_picktime_list = []
    S_net_list = []
    S_sta_list = []
    S_chan_list = []
    S_stalat_list = []
    S_stalon_list = []
    S_staelev_list = []
        
    pick_file = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/event_pick_files/' + str(event_ID) + '.pick', dtype = 'U')
    # print(pick_file)
    
    num_rows = np.arange(len(pick_file))
    # print(num_rows)
    
    for num_row in num_rows:
        
        # print('-------------------------------')
        pick_row = pick_file[num_row]
        # print(pick_row)
        
        pick_time = pick_row[0]
        net = pick_row[1]
        sta = pick_row[2]
        chan = pick_row[3]
        
        # print(chan[2])
        
        # print(pick_time)
        # print(net)
        # print(sta)
        # print(chan)
        
        # Picks on vertical - P-wave arrival
        
        if chan[2] == 'Z':
            
            # print('Vertical channel')
            
            if net == 'CI':
            
                i = np.where(CI_sta_list_array == sta)[0]
                # print(len(i))
                
                if len(i) > 0:
                    
                    # print(CI_info_array[i][0])
                    # print(CI_info_array[i][0][0])
                    # print(CI_info_array[i][0][1])
                    # print(CI_info_array[i][0][2])
                    # print(CI_info_array[i][0][3])
                    
                    P_eventID_list.append(event_ID)
                    P_picktime_list.append(pick_time)
                    P_net_list.append(net)
                    P_sta_list.append(sta)
                    P_chan_list.append(chan)
                    P_stalat_list.append(CI_info_array[i][0][1])
                    P_stalon_list.append(CI_info_array[i][0][2])
                    P_staelev_list.append(CI_info_array[i][0][3])

                else:
                    pass
                    # print("CI Z error:")
                    # print(sta)
            
            elif net == 'PB':
                
                i = np.where(PB_sta_list_array == sta)[0]
                # print(len(i))
                
                if len(i) > 0:
                    
                    # print(PB_info_array[i][0])
                    
                    P_eventID_list.append(event_ID)
                    P_picktime_list.append(pick_time)
                    P_net_list.append(net)
                    P_sta_list.append(sta)
                    P_chan_list.append(chan)
                    P_stalat_list.append(PB_info_array[i][0][1])
                    P_stalon_list.append(PB_info_array[i][0][2])
                    P_staelev_list.append(PB_info_array[i][0][3])
                
                else:
                    pass
                    # print("PB Z error:")
                    # print(sta)
                
            else:
                # print('Not CI or PB')
                pass
        
        # Picks on horizontal - S-wave arrival
        
        if chan[2] == 'N' or chan[2] == 'E' or chan[2] == '1' or chan[2] == '2':
            
            # print('Horizontal channel')
        
            if net == 'CI':
                
                i = np.where(CI_sta_list_array == sta)[0]
                # print(i)
                
                if len(i) > 0:
                    
                    # print(CI_info_array[i][0])
                    
                    S_eventID_list.append(event_ID)
                    S_picktime_list.append(pick_time)
                    S_net_list.append(net)
                    S_sta_list.append(sta)
                    S_chan_list.append(chan)
                    S_stalat_list.append(CI_info_array[i][0][1])
                    S_stalon_list.append(CI_info_array[i][0][2])
                    S_staelev_list.append(CI_info_array[i][0][3])
                
                else:
                    pass
                    # print("CI H error:")
                    # print(sta)
            
            elif net == 'PB':
                
                i = np.where(PB_sta_list_array == sta)[0]
                # print(i)
                
                if len(i) > 0:
                    
                    # print(PB_info_array[i][0])
                    
                    S_eventID_list.append(event_ID)
                    S_picktime_list.append(pick_time)
                    S_net_list.append(net)
                    S_sta_list.append(sta)
                    S_chan_list.append(chan)
                    S_stalat_list.append(PB_info_array[i][0][1])
                    S_stalon_list.append(PB_info_array[i][0][2])
                    S_staelev_list.append(PB_info_array[i][0][3])
                
                else:
                    pass
                    # print("PB H error:")
                    # print(sta)
                               
            else:
                pass
                # print('Not CI or PB')

    print('-------------------------')
    print('EVENT ' + str(event_ID))
    print('-------------------------')

    P_array = np.column_stack((np.array(P_eventID_list), np.array(P_picktime_list), np.array(P_net_list), np.array(P_sta_list), np.array(P_chan_list), np.array(P_stalat_list), np.array(P_stalon_list), np.array(P_staelev_list)))
    # print(P_array)
    
    if len(P_array) == 0:
        p += 1
        
    print('P-wave picks: ' + str(len(P_array)))
    # print(P_array[0])
    np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/picks_w_sta_locs/P_picks/' + str(event_ID) + '.npy', P_array) # LAP
    
    S_array = np.column_stack((np.array(S_eventID_list), np.array(S_picktime_list), np.array(S_net_list), np.array(S_sta_list), np.array(S_chan_list), np.array(S_stalat_list), np.array(S_stalon_list), np.array(S_staelev_list)))
    # print(S_array)
    
    if len(S_array) == 0:
        s += 1
        
    print('S-wave picks: ' + str(len(S_array)))
    # print(S_array[0])
    np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/picks_w_sta_locs/S_picks/' + str(event_ID) + '.npy', S_array) # LAP
    
    if len(P_array) == 0 and len(S_array) == 0:
        z += 1
    
    # print(' ')
    
print(z)
print(p)
print(s)










