#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:33:03 2021

@author: sydneydybing
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from obspy.geodetics import gps2dist_azimuth

results = np.load('testing_for_analysis.npy')
print(len(results))
# results = results[0:10]

# waveforms = 

# w = open('analysis_w_hypdist_result.csv', 'w')

# print(results)

# for i in range(len(results)):
    
#     print(i)
    
#     if results[i][0] == 'nan':
#         rupture = 'nan'
#         station = 'nan'
#         mag = 'nan'
#         pgd = 'nan'
        
#         if results[i][3] == 'true pos': # predicted 1, target 1
#             result = 'true_pos'
        
#         elif results[i][3] == 'true neg': # predicted 0, target 0
#             result = 'true_neg'
            
#         elif results[i][3] == 'false pos': # predicted 1, target 0
#             result = 'false_pos'
        
#         elif results[i][3] == 'false neg': # predicted 0, target 1
#             result = 'false_neg'
        
#         eq_lat = 'nan'
#         eq_lon = 'nan'
#         eq_depth = 'nan'
#         sta_lat = 'nan'
#         sta_lon = 'nan'
#         dist_m = 'nan'
#         line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(rupture,station,mag,pgd,result,eq_lat,eq_lon,eq_depth,sta_lat,sta_lon,dist_m)
#         # print(line)
#         w.write(line)
    
#     else:
        
#         # Getting rupture name and calculating hypocentral distance
        
#         rupture = results[i][0]
#         # print(' ')
#         # print('-------------------------------------------------------------')
#         # print('Rupture = ' + rupture)
#         # print('-------------------------------------------------------------')
    
#         # log = glob('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/all_rupts/' + rupture + '.log') # for laptop
#         log = glob('/hdd/rc_fq/nd3/nd3/output/ruptures/' + rupture + '.log') # for Valdivia
    
#         f = open(log[0],'r')
#         line = f.readlines()
        
#         # Getting hypocenter location
#         hyp_loc_junk = line[16]
#         eq_lon = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[0])
#         eq_lat = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[1])
#         eq_depth = float(hyp_loc_junk.split(' ')[2].split('(')[1].split(')')[0].split(',')[2])
#         eq_depth_m = eq_depth * 1000
        
#         station = results[i][1]
#         # print(station)
        
#         # Getting PGD from the summary files
        
#         # print(rupture)
#         # rupt_num = rupture[4:]
#         # print(rupt_num)
        
#         summary_file_path = '/hdd/rc_fq/nd3/nd3/output/waveforms/' + rupture + '/_summary.' + rupture + '.txt'
#         # print(summary_file_path)
        
#         summary_file_stations = np.genfromtxt(summary_file_path, skip_header=1, dtype = str, usecols = 0)
        
#         j = np.where(summary_file_stations == station)[0]
        
#         summary_file = np.genfromtxt(summary_file_path, skip_header=1, dtype = str)
#         # print(summary_file)
#         pgd = float(summary_file[j][0][6])
#         # print(station)
#         # print(summary_file[j][0][0])
#         # print(pgd)
        
#         station_names = np.genfromtxt('/home/sdybing/GNSS_project/newtrain_march/rc_grid.gflist', dtype = 'U', usecols = 0)
#         # print(station_names)
        
#         k = np.where(station_names == station)[0]
#         # print(k)
        
#         # Getting station location
        
#         station_info = np.genfromtxt('/home/sdybing/GNSS_project/newtrain_march/rc_grid.gflist', dtype = 'U')
#         sta_lon = float(station_info[0][1])
#         sta_lat = float(station_info[0][2])

#         # Calculating hypocentral distance

#         distaz = gps2dist_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)
#         dist_m = distaz[0]
        
#         # Adding info about result of training
        
#         if results[i][3] == 'true pos': # predicted 1, target 1
#             mag = float(results[i][2])
#             dot = 2
#             result = 'true_pos'
        
#         elif results[i][3] == 'true neg': # predicted 0, target 0
#             mag = float(results[i][2])
#             dot = 1
#             result = 'true_neg'
            
#         elif results[i][3] == 'false pos': # predicted 1, target 0
#             mag = float(results[i][2])
#             dot = 0
#             result = 'false_pos'
        
#         elif results[i][3] == 'false neg': # predicted 0, target 1
#             mag = float(results[i][2])
#             dot = -1
#             result = 'false_neg'
            
#         else:
#             pass
        
#         line = '%s\t%s\t%.2f\t%.6f\t%s\t%.6f\t%.6f\t%.2f\t%.6f\t%.6f\t%.3f\n'%(rupture,station,mag,pgd,result,eq_lat,eq_lon,eq_depth,sta_lat,sta_lon,dist_m)
#         # print(line)
#         w.write(line)

# w.close()





        