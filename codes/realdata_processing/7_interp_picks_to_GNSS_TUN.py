#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:29:28 2022

@author: sydneydybing
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator
import matplotlib.pyplot as plt

event_catalog = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/events.txt', dtype = 'U')
# print(event_catalog)

for kevent in range(len(event_catalog)):

    event_ID = event_catalog[kevent,0]
    mag = event_catalog[kevent,5]

    try:        
        
        print('')
        print('----------------------')
        print('Event ' + str(event_ID) + ': P-waves')
        print('----------------------')
        
        P_array = np.load('/hdd/Ridgecrest/eq_days_split/seismic_traveltimes/P_waves/' + str(event_ID) + '.npy')
        # print(P_array)
        
        ###### Interpolate P-wave pick times ######
        
        P_latitude = np.asfarray(P_array[:,2])
        P_longitude = np.asfarray(P_array[:,3])
        P_travel_times = np.asfarray(P_array[:,4])
        
        P_interp_lat = np.linspace(min(P_latitude), max(P_latitude), num = 1000)
        P_interp_lon = np.linspace(min(P_longitude), max(P_longitude), num = 1000)
        
        P_interp_lon, P_interp_lat = np.meshgrid(P_interp_lon, P_interp_lat)
        
        P_interp = LinearNDInterpolator(list(zip(P_longitude, P_latitude)), P_travel_times)
        
        P_interp_tt = P_interp(P_interp_lon, P_interp_lat)
        
        plt.pcolormesh(P_interp_lon, P_interp_lat, P_interp_tt, shading = 'auto')
        plt.plot(P_longitude, P_latitude, '.k', label = 'Seismic station')
        plt.legend()
        plt.colorbar(label = 'Travel time (s)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('P-wave travel times: event ID ' + str(event_ID) + ' (magnitude ' + str(mag) + ')')
        plt.savefig('/hdd/Ridgecrest/eq_days_split/seismic_traveltimes/interpolation_plots/P_waves/' + str(event_ID) + '.png', format = 'PNG')
        plt.close()
        
        ### Find values for GNSS stations ###
        
        GNSS_stas = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/GNSS_stas.txt', dtype = 'U')
        # print(GNSS_stas)
        
        GNSS_stacodes = GNSS_stas[:,2]
        
        GNSS_P_tts = []
        GNSS_P_stas = []
        P_event_IDs = []
        
        for kGsta in range(len(GNSS_stacodes)):
            
            P_dists = []
            lon_list = []
            lat_list = []
            
            GNSS_stacode = GNSS_stas[kGsta,2]
            print(GNSS_stacode)
            
            GNSS_lon = float(GNSS_stas[kGsta,0])
            GNSS_lat = float(GNSS_stas[kGsta,1])
            
            print('GNSS station location: (' + str(GNSS_lat) + ', ' + str(GNSS_lon) + ')')
            
            for i_lon in P_interp_lon[0]:
            
                # print(i_lon)
                
                for i_lat in P_interp_lat[:,0]:
                    
                    # print(i_lat)
            
                    dist = np.sqrt((GNSS_lon - i_lon)**2 + (GNSS_lat - i_lat)**2)
            
                    P_dists.append(dist)
                    lon_list.append(i_lon)
                    lat_list.append(i_lat)
                  
            P_dists = np.array(P_dists)
            
            lon_array = np.array(lon_list)
            
            lat_array = np.array(lat_list)
                
            i = np.argmin(P_dists)
            
            print('Distance (km) to closest interpolated point: ' + str(P_dists[i] * 111))
            print('Interpolated point location: (' + str(lat_array[i]) + ', ' + str(lon_array[i]) + ')')
            
            a = np.where(P_interp_lon[0] == lon_array[i])[0]
            b = np.where(P_interp_lat[:,0] == lat_array[i])[0]
            
            GNSS_time = P_interp_tt[b,a][0]
            
            print('Interpolated GNSS travel time: ' + str(GNSS_time))
            
            GNSS_P_tts.append(GNSS_time)
            GNSS_P_stas.append(GNSS_stacode)
            P_event_IDs.append(event_ID)
        
        # print(GNSS_P_tts)    
        # print(GNSS_P_stas)
        
        GNSS_P_array = np.column_stack((np.array(P_event_IDs), np.array(GNSS_P_stas), np.array(GNSS_P_tts)))
        
        np.save('/hdd/Ridgecrest/eq_days_split/GNSS_traveltimes/P_waves/' + str(event_ID) + '.npy', GNSS_P_array) # TUN
        
    except:
        
        print('P-wave error: event ID ' + str(event_ID))
        nan_P_array = np.column_stack(('nan', 'nan', 'nan'))
        np.save('/hdd/Ridgecrest/eq_days_split/GNSS_traveltimes/P_waves/' + str(event_ID) + '.npy', nan_P_array) # TUN
        
    try:        
        
        print('')
        print('----------------------')
        print('Event ' + str(event_ID) + ': S-waves')
        print('----------------------')
        
        S_array = np.load('/hdd/Ridgecrest/eq_days_split/seismic_traveltimes/S_waves/' + str(event_ID) + '.npy')
        # print(S_array)
        
        ###### Interpolate S-wave pick times ######
        
        S_latitude = np.asfarray(S_array[:,2])
        S_longitude = np.asfarray(S_array[:,3])
        S_travel_times = np.asfarray(S_array[:,4])
        
        S_interp_lat = np.linspace(min(S_latitude), max(S_latitude), num = 1000)
        S_interp_lon = np.linspace(min(S_longitude), max(S_longitude), num = 1000)
        
        S_interp_lon, S_interp_lat = np.meshgrid(S_interp_lon, S_interp_lat)
        
        S_interp = LinearNDInterpolator(list(zip(S_longitude, S_latitude)), S_travel_times)
        
        S_interp_tt = S_interp(S_interp_lon, S_interp_lat)
        
        plt.pcolormesh(S_interp_lon, S_interp_lat, S_interp_tt, shading = 'auto')
        plt.plot(S_longitude, S_latitude, '.k', label = 'Seismic station')
        plt.legend()
        plt.colorbar(label = 'Travel time (s)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('S-wave travel times: event ID ' + str(event_ID) + ' (magnitude ' + str(mag) + ')')
        plt.savefig('/hdd/Ridgecrest/eq_days_split/seismic_traveltimes/interpolation_plots/S_waves/' + str(event_ID) + '.png', format = 'PNG')
        plt.close()
        
        ### Find values for GNSS stations ###
        
        GNSS_stas = np.genfromtxt('/hdd/Ridgecrest/eq_days_split/GNSS_stas.txt', dtype = 'U')
        # print(GNSS_stas)
        
        GNSS_stacodes = GNSS_stas[:,2]
        
        GNSS_S_tts = []
        GNSS_S_stas = []
        S_event_IDs = []
        
        for kGsta in range(len(GNSS_stacodes)):
            
            S_dists = []
            lon_list = []
            lat_list = []
            
            GNSS_stacode = GNSS_stas[kGsta,2]
            print(GNSS_stacode)
            
            GNSS_lon = float(GNSS_stas[kGsta,0])
            GNSS_lat = float(GNSS_stas[kGsta,1])
            
            print('GNSS station location: (' + str(GNSS_lat) + ', ' + str(GNSS_lon) + ')')
            
            for i_lon in S_interp_lon[0]:
            
                # print(i_lon)
                
                for i_lat in S_interp_lat[:,0]:
                    
                    # print(i_lat)
            
                    dist = np.sqrt((GNSS_lon - i_lon)**2 + (GNSS_lat - i_lat)**2)
            
                    S_dists.append(dist)
                    lon_list.append(i_lon)
                    lat_list.append(i_lat)
                  
            S_dists = np.array(S_dists)
            
            lon_array = np.array(lon_list)
            
            lat_array = np.array(lat_list)
                
            i = np.argmin(S_dists)
            
            print('Distance (km) to closest interpolated point: ' + str(S_dists[i] * 111))
            print('Interpolated point location: (' + str(lat_array[i]) + ', ' + str(lon_array[i]) + ')')
            
            a = np.where(S_interp_lon[0] == lon_array[i])[0]
            b = np.where(S_interp_lat[:,0] == lat_array[i])[0]
            
            GNSS_time = S_interp_tt[b,a][0]
            
            print('Interpolated GNSS travel time: ' + str(GNSS_time))
            
            GNSS_S_tts.append(GNSS_time)
            GNSS_S_stas.append(GNSS_stacode)
            S_event_IDs.append(event_ID)
        
        # print(GNSS_S_tts)    
        # print(GNSS_S_stas)
        
        GNSS_S_array = np.column_stack((np.array(S_event_IDs), np.array(GNSS_S_stas), np.array(GNSS_S_tts)))
        
        np.save('/hdd/Ridgecrest/eq_days_split/GNSS_traveltimes/S_waves/' + str(event_ID) + '.npy', GNSS_S_array) # TUN
        
    except:
        
        print('S-wave error: event ID ' + str(event_ID))
        nan_S_array = np.column_stack(('nan', 'nan', 'nan'))
        np.save('/hdd/Ridgecrest/eq_days_split/GNSS_traveltimes/S_waves/' + str(event_ID) + '.npy', nan_S_array) # TUN
    
    
    