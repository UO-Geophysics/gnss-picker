#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:47:03 2021

@author: sydneydybing
"""
from mudpy import fakequakes,runslip,forward,viewFQ
import numpy as np
from obspy.core import UTCDateTime,read
from shutil import copy

########                            GLOBALS                             ########

home = '/hdd/rc_fq/nd3/' # set up for Krakatoa or Valdivia
project_name = 'nd3'
run_name = 'nd3'

################################################################################


##############             What do you want to do??           ##################

init = 0
make_ruptures = 0
make_GFs = 0
make_synthetics = 0
make_waveforms = 1
make_hf_waveforms = 0
match_filter = 0
make_statics = 0
# Things that only need to be done once
load_distances = 1 # for make_ruptures
G_from_file = 0 # for make_waveforms

###############################################################################

##############                 Run parameters                ##################

# Runtime parameters 
ncpus = 50                                        # How many CPUS you want to use for parallelization (needs to be at least 2)
Nrealizations = 2800                                # Number of fake ruptures to generate per magnitude bin - ncups overrides this?
hot_start = 0

# File parameters
model_name = 'mojave.mod'
# model_name = 'ridgecrest.mod'                      # Velocity model file name
fault_name = 'ridgecrest_m7_fault3.fault'                    # Fault model name
mean_slip_name = None                            # Set to path of .rupt file if patterning synthetic runs after a mean rupture model
# run_name = 'rcrest_m7'                            # Base name of each synthetic run (i.e. mentawai.000000, mentawai.000001, etc...)
rupture_list = 'ruptures.list'                   # Name of list of ruptures that are used to generate waveforms.  'ruptures.list' uses the full list of ruptures FakeQuakes creates. If you create file with a sublist of ruptures, use that file name.
distances_name = 'new_distrib'                      # Name of matrix with estimated distances between subfaults i and j for every subfault pair
# load_distances = 0                               # This should be zero the first time you run FakeQuakes with your fault model.

# Source parameters
UTM_zone = '11S'                                 # UTM_zone for rupture region 
time_epi = UTCDateTime('2019-07-06T03:19:53.040')   # Origin time of event (can set to any time, as long as it's not in the future)
# target_Mw = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])                      # Desired magnitude(s), can either be one value or an array
target_Mw_flip = np.arange(6.5,7.4,0.1)
target_Mw = np.flip(target_Mw_flip)
hypocenter = None                                # Coordinates of subfault closest to desired hypocenter, or set to None for random
force_hypocenter = False                         # Set to True if hypocenter specified
rake = 180                                        # Average rake for subfaults
scaling_law = 'S'                                # Type of rupture: T for thrust, S for strike-slip, N for normal
force_magnitude = False                          # Set to True if you want the rupture magnitude to equal the exact target magnitude
force_area = False                               # Set to True if you want the ruptures to fill the whole fault model

# Correlation function parameters
hurst = 0.4                                      # Hurst exponent form Melgar and Hayes 2019
Ldip = 'auto'                                    # Correlation length scaling: 'auto' uses Melgar and Hayes 2019, 'MB2002' uses Mai and Beroza 2002
Lstrike = 'auto'                                 # Same as above
slip_standard_deviation = 0.9                    # Standard deviation for slip statistics: Keep this at 0.9
lognormal = True                                 # Keep this as True to solve the problem of some negative slip subfaults that are produced

# Rupture propagation parameters
rise_time_depths = [1,2]                       # Transition depths for rise time scaling (if slip shallower than first index, rise times are twice as long as calculated)
max_slip = 40                                    # Maximum sip (m) allowed in the model
max_slip_rule = False                            # If true, uses a magntidude-depence for max slip
shear_wave_fraction_shallow = 0.5                        # 0.8 is a standard value (Mai and Beroza 2002)
shear_wave_fraction_deep = 0.8
source_time_function = 'dreger'                  # options are 'triangle' or 'cosine' or 'dreger'
stf_falloff_rate = 4                             # Only affects Dreger STF, 4-8 are reasonable values
num_modes = 72                                   # Number of modes in K-L expansion
slab_name = None                                 # Slab 2.0 Ascii file for 3D geometry, set to None for simple 2D geometry
mesh_name = None                                 # GMSH output file for 3D geometry, set to None for simple 2D geometry

# Green's Functions parameters
GF_list = 'rc_grid.gflist'                            # Stations file name
G_name = 'rc'                                    # Basename you want for the Green's functions matrices
# make_GFs = 0                                     # This should be 1 to run Green's functions
# make_synthetics = 0                              # This should be 1 to make the synthetics
# G_from_file = 0                                  # This should be zero the first time you run FakeQuakes with your fault model and stations.

# fk parameters
# used to solve wave equation in frequency domain 
dk = 0.1 ; pmin = 0 ; pmax = 1 ; kmax = 20             # Should be set to 0.1, 0, 1, 20
custom_stf = None                                # Assumes specified source time function above if set to None

# Low frequency waveform parameters
dt = 1.0                                         # Sampling interval of LF data 
NFFT = 512                                       # Number of samples in LF waveforms (should be in powers of 2)
# dt*NFFT  =  length of low-frequency dispalcement record
# want this value to be close to duration (length of high-frequency record)

# High frequency waveform parameters
stress_parameter = 50                            # Stress drop measured in bars (standard value is 50)
moho_depth_in_km = 30.0                          # Average depth to Moho in this region 
Pwave = True                                     # Calculates P-waves as well as S-waves if set to True, else just S-Waves
kappa = None                                     # Station kappa values: Options are GF_list for station-specific kappa, a singular value for all stations, or the default 0.04s for every station if set to None
hf_dt = 0.01                                     # Sampling interval of HF data
duration = 250                                   # Duration (in seconds) of HF record

high_stress_depth = 30                           # Doesn't do anything, but still shows up as a parameter. Set to whatever you want. 

# Match filter parameters
zero_phase = True                                # If True, filters waveforms twice to remove phase, else filters once
order = 4                                        # Number of poles for filters
fcorner = 0.998                                  # Corner frequency at which to filter waveforms (needs to be between 0 and the Nyquist frequency)

###############################################################################

# Set up project folder

if init == 1:
    
    fakequakes.init(home,project_name)

    # Copy files into the project folder

    copy('/hdd/rc_fq/nd3/mojave.mod', '/' + home + '/' + project_name + '/structure')
    copy('/hdd/rc_fq/nd3/ridgecrest_m7_fault3.fault', '/' + home + '/' + project_name + '/data/model_info')
    copy('/hdd/rc_fq/nd3/rc_grid.gflist', '/' + home + '/' + project_name + '/data/station_info')
    
# Generate rupture models

if make_ruptures == 1:
    
    fakequakes.generate_ruptures(home,project_name,run_name,fault_name,slab_name,mesh_name,load_distances,
        distances_name,UTM_zone,target_Mw,model_name,hurst,Ldip,Lstrike,num_modes,Nrealizations,rake,
        rise_time_depths,time_epi,max_slip,source_time_function,lognormal,slip_standard_deviation,scaling_law,
        ncpus,mean_slip_name=mean_slip_name,force_magnitude=force_magnitude,force_area=force_area,
        hypocenter=hypocenter,force_hypocenter=force_hypocenter,shear_wave_fraction_shallow=shear_wave_fraction_shallow,
        shear_wave_fraction_deep=shear_wave_fraction_deep,max_slip_rule=max_slip_rule)
    
# Make Green's functions

if make_GFs == 1 or make_synthetics == 1:
    
    runslip.inversionGFs(home,project_name,GF_list,None,fault_name,model_name,
        dt,None,NFFT,None,make_GFs,make_synthetics,dk,pmin,
        pmax,kmax,0,time_epi,hot_start,ncpus,custom_stf,impulse=True) 
    
# # Make low frequency displacement waveforms

if make_waveforms == 1:
    
    forward.waveforms_fakequakes(home,project_name,fault_name,rupture_list,GF_list, # need to shorten path again
        model_name,run_name,dt,NFFT,G_from_file,G_name,source_time_function,
        stf_falloff_rate,ncpus=ncpus)
    
# See some of the waveforms

# stas = np.genfromtxt('/Users/sydneydybing/RC_FQ/flt3_tst_rnge/data/station_info/RC_gflist_short.gflist', usecols=0, dtype=str)
# rupt = '/rcrest_M6.000000'

# for sta in stas:
    
#     st = read(f'{home}{project_name}/output/waveforms' + rupt + '/' + sta + '.LYE.sac')
#     st.plot()



