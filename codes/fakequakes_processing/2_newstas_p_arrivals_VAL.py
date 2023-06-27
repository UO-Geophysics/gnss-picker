#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:08:08 2021

@author: sydneydybing
"""
import numpy as np
from obspy.core import UTCDateTime
from obspy import read
from numpy import genfromtxt,where,arange,ones,zeros,array,tile,argmin
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from glob import glob

### FOR VALDIVIA HDD ###

ruptures = np.genfromtxt('/hdd/rc_fq/nd3/ruptures.txt',dtype='str')
test_ruptures = ['rupt.000000', 'rupt.000001', 'rupt.000002', 'rupt.000003' ,'rupt.000004']
    
# w = open('/home/sdybing/rc_fq/test_p_arrivals.csv', 'w') # saves arrivals for every event in one csv

for rupture in ruptures:
    
    # w = open('/home/sdybing/rc_fq/npd_p_arrivals/arrival_times_' + rupture + '.csv', 'w')
    w = open('/hdd/rc_fq/nd3/p_arrivals/allstas_arrival_times_' + rupture + '.csv', 'w') # saves arrivals for each event in a separate csv
    
    print('-------------------------------------------------------------')
    print('Rupture = ' + rupture)
    print('-------------------------------------------------------------')
    print(' ')
    
    log = glob('/hdd/rc_fq/nd3/nd3/output/ruptures/' + rupture + '.log')
    
    f = open(log[0],'r')
    line = f.readlines()
    
    # getting hypocenter location
    hyp_loc_junk = line[16]
    hyp_loc_1 = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[0])
    hyp_loc_2 = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[1])
    hyp_loc_3 = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[2])
    
    epicenter = []
    epicenter.append(hyp_loc_1)
    epicenter.append(hyp_loc_2)
    epicenter.append(hyp_loc_3)
    # print(epicenter)
    
    # getting hypocenter time
    hyp_time_junk = line[17]
    hyp_time = hyp_time_junk.split(' ')[2].split('Z')[0]
    
    time_epi = UTCDateTime(hyp_time)  
    
    # stations = np.genfromtxt('/home/sdybing/rc_fq/stations_5cm_PGD.txt', dtype = 'str') # stations with 5 cm PGD for a M7
    stations = np.genfromtxt('/hdd/rc_fq/nd3/stas_alone.txt', dtype = 'str')
    
    Nsta = len(stations) # number of stations
    
    rootpath = '/hdd/rc_fq/nd3/nd3/output/waveforms/' + rupture + '/' # waveform file locations
    
    lonlat = genfromtxt('/hdd/rc_fq/nd3/stas_info.txt', usecols=[0,1]) # locations of stations
    # print(lonlat)
    
    station_catalogue = genfromtxt('/hdd/rc_fq/nd3/stas_info.txt', usecols=[2], dtype='U') # list of stations
    
    velmod = TauPyModel(model='/hdd/rc_fq/nd3/mojave.npz') # velocity model
    
    predictedP = 9999*ones(len(stations))
    
    # Get predicted arrivals
    
    for ksta in range(len(stations)):
        
        # Find station coordinates
        i = where(station_catalogue == stations[ksta])[0]
        lon_sta = lonlat[i,0]
        lat_sta = lonlat[i,1]
                        
        deg = locations2degrees(lon_sta, lat_sta, epicenter[0], epicenter[1]) # calculates distance between station loc and epicenter loc
        arrivals = velmod.get_travel_times(source_depth_in_km = epicenter[2], distance_in_degree = deg,phase_list = ['P','Pn','p'])
        
        # Determine P and S arrivals
        for kphase in range(len(arrivals)):
            if 'P' == arrivals[kphase].name or 'p' == arrivals[kphase].name or 'Pn' == arrivals[kphase].name:
                if arrivals[kphase].time < predictedP[ksta]:
                    predictedP[ksta] = arrivals[kphase].time
                    
        print('Station ' + stations[ksta] + ': predicted arrival = ' + str(time_epi + predictedP[ksta]))
        
        line = '%s\t%s\t%s\n'%(rupture,str(stations[ksta]),str(time_epi + predictedP[ksta]))
        # print(line)
        w.write(line)
        
    print(' ')
    
    w.close()

# w.close()














