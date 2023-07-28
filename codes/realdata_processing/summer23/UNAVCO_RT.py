#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from obspy import UTCDateTime
import numpy as np
from obspy import Stream,Trace

def convertSec( n ):
    n = float(n)
    return ( "%06.0f"%n )

def convertLat( n ):
    n = str(n)
    return( float(n[0:2]) + float(n[2:13])/60.0 )

def convertLong( n ):
    n = str(n)
    return(  float(n[0:3]) + float(n[3:14])/60.0  )

def make_UTCDateTime(date,time_of_day):
    
    #yyyy-mm-dd
    date = str(date)
    
    #get hour min sec
    hour = str(time_of_day)[0:2]
    minute = str(time_of_day)[2:4]
    second = str(time_of_day)[4:6]
    
    utc_object = UTCDateTime(date+'T'+hour+':'+minute+':'+second)
    return(utc_object)

def LongitudeDirection2sign( n ):
    if n=='W':
        sign=-1
    else:
        sign=1
    return sign

def LatitudeDirection2sign( n ):
    if n=='S':
        sign=-1
    else:
        sign=1
    return sign


def parse(file):

    # read the file
    df = pd.read_csv (  file,   names = [ 'NMEA1','NMEA2','Time', 'Date', 'Latitude','LatDirection','Longitude','LongDirection', 'PositionType', 'NumberSat', 'PDOP', 'Height','HeightUnits','nFields','EastSigma','NorthSigma','LastField' ], error_bad_lines=False, low_memory=False   )

    # some clean up, remove line with NaN and that do not start with PTNL
    df = df.dropna()
    df = df[df['NMEA1'].str.startswith('$PTNL')]

    # seperate out the UpSigma and Checksum from last field
    df[[ 'UpSigma','CheckSum' ]] = df.LastField.str.split("*",expand=True)

    # some reformatting
    df[ 'Date']        = pd.to_datetime(df['Date'] , format='%m%d%y' ).dt.date
    df[ 'Time']        = df.apply(lambda x : convertSec( x['Time'] ) ,axis=1) 
    df[ 'UTCDateTime'] = df.apply(lambda x : make_UTCDateTime(x['Date'],x['Time'] ) ,axis=1) 
    df[ 'Seconds_of_day'] = df['UTCDateTime'] - df['UTCDateTime'][0]
    df[ 'Latitude']    = df.apply(lambda x : convertLat( x['Latitude'] ) ,axis=1) 
    df[ 'Longitude']   = df.apply(lambda x : convertLong( x['Longitude'] ) ,axis=1) 
    df[ 'Height']      = np.array(df['Height'].str.replace('EHT', ''),dtype='float')
    
    #apply correct longitude sign
    df['LonSign']        = df.apply(lambda x : LongitudeDirection2sign( x['LongDirection'] ) ,axis=1) 
    df['Longitude'] = df.LonSign*df.Longitude
    
    #apply correct latitude sign
    df['LatSign']        = df.apply(lambda x : LatitudeDirection2sign( x['LatDirection'] ) ,axis=1) 
    df['Latitude'] = df.LatSign*df.Latitude

    #remove some unneeded fields`
    df.drop(['LastField'],axis=1, inplace=True)  
    df.drop(['nFields'],axis=1, inplace=True)  
    df.drop(['NMEA1'],axis=1, inplace=True)  
    df.drop(['NMEA2'],axis=1, inplace=True)  
    df.drop(['CheckSum'],axis=1, inplace=True)  

    return(df)

def unavco2neu(df):

    # WGS84 stuff
    
    a = 6378137.0 # semimajor axis
    f = 1/298.257223563 # flattening
    es = 2*f-f**2 # ellipsoid eccentricity squared
    lat = np.array(np.deg2rad(df.Latitude))
    lon = np.array(np.deg2rad(df.Longitude))
    h = np.array(df.Height)
    
    # Make eta
    
    eta = a/(1-es*np.sin(lat))**(1/2)
    
    # Convert to ECEF (earth centered, earth fixed)
    
    x = (eta + h) * np.cos(lat) * np.cos(lon)
    y = (eta + h) * np.cos(lat) * np.sin(lon)
    z = (eta * (1-es) + h) * np.sin(lat)
    
    # Convert to ENU
    # Initial coordinates
    
    x0 = x[0]
    y0 = y[0]
    z0 = z[0]
    
    lat0 = lat[0]
    lon0 = lon[0]
    
    # Make rotation matrix
    
    R = np.array([[-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                 [-np.sin(lon0), np.cos(lon0), 0],
                 [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
    
    # print(R)
    
    # Concatenate into long matrix
    
    x = np.expand_dims(x,0) - x0
    y = np.expand_dims(y,0) - y0
    z = np.expand_dims(z,0) - z0
    
    xyz = np.r_[x,y,z]
    # print(xyz.shape)
    
    # Now rotate into local NEU
    
    neu = R.dot(xyz)
    # print(neu)
    
    n = neu[0,:]
    e = neu[1,:]
    u = neu[2,:]

    return n,e,u

def pandas2obspy(df,n,e,u):
    
    # Put in a stream function

    stn = Stream()
    ste = Stream()
    stu = Stream()
    
    # Find the gaps using the differences in seconds of the day
    # Count number of gaps (number of traces)
    
    gaps = np.diff(df.Seconds_of_day)
    gaps = np.where(gaps>1)[0] # rows to say only interested in rows. POsitions where gaps are found
    Ngaps = len(gaps)
    Ntraces = Ngaps + 1
    
    # Initialize the trace start/end counters
    
    istart = 0
    iend = gaps[0]
    
    for k in range(Ntraces):
        
        # print(istart)
        # print(iend)
        # print('----')
        
        stn += Trace()
        ste += Trace()
        stu += Trace()
        
        # Add the data
        
        stn[k].data = n[istart:iend]
        ste[k].data = e[istart:iend]
        stu[k].data = u[istart:iend]
    
        # Fix the dates
        
        stn[k].stats.starttime = df.UTCDateTime[istart] # wrong - needs debugging in UNAVCO_RT date function
        ste[k].stats.starttime = df.UTCDateTime[istart]
        stu[k].stats.starttime = df.UTCDateTime[istart]
    
        # Update the counters
        
        if k == Ngaps - 1: # the last gap
            istart = iend + 1
            iend = -1
        
        elif k == Ngaps: # you've reached the end and AVOID updating the counters
            pass
        
        else: # everything else
            istart = iend + 1
            iend = gaps[k+1]
        
    return stn,ste,stu        
        
        
        
        
        
        
