#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:52:25 2022

@author: sydneydybing
"""

from os import path,makedirs
from obspy.clients.fdsn import Client
import obspy.io.quakeml.core as quakeml
# from obspy.core import quakeml
from obspy.core.event import read_events
from obspy import UTCDateTime
from numpy import zeros,genfromtxt
from matplotlib import pyplot as plt
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader


################         What do you want to do?            ###################

get_catalog = True
make_events_file = True
make_pick_files = True
mass_download = False

###############################################################################


'''
# You can avoid running the catalog search from here and also use the USGS search
# https://earthquake.usgs.gov/earthquakes/search/ which explores ComCat and can
# output both text and QuakeML
# '''

# #Time period for search (A couple of days)
# tstart = UTCDateTime("2019-07-02")
# tend = UTCDateTime("2019-07-07")

# # geographic Box for search (socal)
# lon_query = [-118.5,-116.5]
# lat_query = [35,36.5]  

# #Which data centers should I query and in what order?
# providers = ['SCEDC']
# maxradius = 5 #in degrees 

# #times before/after origin time to download
# tprevious = 2.0
# tafter = 60.0

# #What data center and where should I save it dofus?
# catalog_file = '/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rcrest_M4.3.quakeml'
events_file = '/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/events.txt'
pick_file_path = '/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/event_pick_files'

# #Where should I save things
# download_path = '/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/waveforms'

# #Event of what size?
# min_magnitude = 4.3

# #Which channels
# channels = ["HNN'", "HNE", "HNZ"]

################################################################################

starttime = UTCDateTime("2019-06-01T00:00:00")
endtime = UTCDateTime("2020-07-30T23:59:59")

if get_catalog:
    
    client = Client("SCEDC")
    catalog = client.get_events(starttime = starttime, endtime = endtime, latitude = 35.7695, longitude = -117.5993333, maxradius = 1, minmagnitude = 4.3)
    print(catalog)
        
# load the catalog
# catalog = read_events(catalog_file)

if make_events_file:
    
    f = open(events_file,'w')
    f.write('# ID,origin_time,lon,lat,depth(km),Mag\n')
    
    for kevent in range(len(catalog)):
        
        ev = catalog[kevent]
        time = ev.origins[0].time
        # ID = ev.resource_id.id.split('=')[-2].split('&')[0]
        long_ID = ev.resource_id
        str_long_ID = str(long_ID)
        # print(str_long_ID)
        ID = str_long_ID.split('=')[1]
        print(ID)
        mag = ev.magnitudes[0].mag
        lat = ev.origins[0].latitude
        lon = ev.origins[0].longitude
        depth = ev.origins[0].depth/1000
        
        #Write the line      
        line = '%s\t%s\t%.4f\t%.4f\t%.2f\t%.2f\n' % (ID,time,lon,lat,depth,mag)
        
        f.write(line)
        
    f.close()
        

if make_pick_files:
   
    events = genfromtxt(events_file,usecols = 0,dtype = 'U')
    print(events)
    pick_client  =  Client('SCEDC')
   
    for kevent in range(len(events)):
        
        id_event = events[kevent]
        
        # if 'ci' in id_event: #SoCla event, get picks, otehrwise ignore
            
        print('Creating pick file for event ' + str(kevent) + ' of ' + str(len(events)))
        
        pick_file = pick_file_path + '/' + id_event + '.pick'
        f = open(pick_file,'w')
        f.write('#pick time, network, station ,channel\n')
        
        id_event = id_event.replace('ci','')
        ev = pick_client.get_events(eventid = id_event, includearrivals = True)
        picks = ev[0].picks
        
        for kpick in range(len(picks)):
            
            pick = picks[kpick]
            time = pick.time
            net = pick.waveform_id.network_code
            sta = pick.waveform_id.station_code
            chan = pick.waveform_id.channel_code
            line = '%s\t%s\t%s\t%s\n' % (time,net,sta,chan)
            f.write(line)
            
        f.close()
            
        
            
            





    
        
        
# if mass_download:
    
#     for kevent in range(len(catalog)):
#         # print '\n######   Working on event %d of %d   ######\n' %(kevent,len(catalog))
#         ev = catalog[kevent]
#         origin_time  =  ev.origins[0]['time']
#         lon = ev.origins[0]['longitude']
#         lat = ev.origins[0]['latitude']
        
#         #Event string for folder
#         ev_string = ev.resource_id.id.split(' = ')[-2].split('&')[0]
#         ev_folder = download_path+'/'+ev_string
#         sta_folder = ev_folder+'/_station_xml/'
        
#         if path.exists(ev_folder) =  = False:  #Path exists, clobber?
#             makedirs(ev_folder)
#             makedirs(sta_folder)
    
    
#         # Circular domain around the epicenter. This will download all data between
#         # 70 and 90 degrees distance from the epicenter. This module also offers
#         # rectangular and global domains. More complex domains can be defined by
#         # inheriting from the Domain class.
#         domain  =  CircularDomain(latitude = lat, longitude = lon, minradius = 0,
#                     maxradius = maxradius)
        
#         restrictions  =  Restrictions(
#             # Get data from 5 minutes before the event to one hour after the
#             # event. This defines the temporal bounds of the waveform data.
#             starttime = origin_time - tprevious,
#             endtime = origin_time + tafter,
#             # You might not want to deal with gaps in the data. If this setting is
#             # True, any trace with a gap/overlap will be discarded.
#             reject_channels_with_gaps = False,
#             # And you might only want waveforms that have data for at least 95 % of
#             # the requested time span. Any trace that is shorter than 95 % of the
#             # desired total duration will be discarded.
#             channel_priorities = channels)
        
#         # No specified providers will result in all known ones being queried.
#         mdl  =  MassDownloader(providers = providers)
#         # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
#         # folders with automatically chosen file names.
#         try:
#             mdl.download(domain, restrictions, mseed_storage = ev_folder,
#                         stationxml_storage = sta_folder)
#         except:
#             print '\n\n#########     DOWNLOAD ERROR: moving on      ##########\n\n'
