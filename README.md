# gnss-picker
Codes for project that identifies earthquake onset in RT-GNSS data

# gnss-picker
Codes for project that identifies earthquake onset in RT-GNSS data

DATA and FIGURES directories not included

Code Workflow

COMPONENTS ORDERED N E Z FOR THIS PROJECT

For FakeQuakes data (codes/fakequakes_processing)

1. m7_newdistrib.fq.py
    1. Makes fakequakes ruptures and waveforms
    2. Requires .mod (velocity model), .fault (fault model), and .glfist (stations) files
2. nd_p_arrivals_VAL.py
    1. Calculates P-wave arrival times for ruptures and stations
    2. Requires station lists with coordinates and with stations alone in .txt format, ruptures list in .txt without .rupts at the end, velocity model as .npz
3. nd_format_data_VAL.py
    1. Formats data for entry into GNSS code (centers on P-wave arrival time, trims, and puts 3 components side by side)
    2. This version does NOT contain the mess for getting the log files for the renamed ruptures. Will need to go into older folders for that.
    3. Requires rupture list, station list, log files and arrival time files, waveforms, and summary files
    4. format_data_bins_VAL.py version limits the data distribution by PGD
4. If you don’t need more than 2.6 million noise samples, use shorten_noise_h5py.py to shorten the 2.6 million long noise array to the desired length to match the length of the data array.

For real data (codes/realdata_processing) - newtrain_march most recent I think? Also use More_RealData folder

1. split_real_data_LAP.py
    1. Splits the daily e, n, u mMSEED files into separate 128s long mSEED files while interpolating for gaps
    2. Requires station list, channel list, list of dates, and the daily mSEED files
2. format_real_data_LAP.py
    1. Transforms the 128s e, n, u mSEED files into the CNN format: channels side by side in each row in an array that’s saved as the hdf5 format. Also saves the metadata array with the station name, date, start time, end time, and counter as an .npy. 
    2. Requires station list, channel list, list of dates, list of ns (counter from previous code) and the split mSEED files
3. normalize_realdata.py
    1. Normalize the real data from realdata_data.hdf5 by subtracting the mean from each component so that it’s more similar to how the Fakequakes are formatted
        1. Produces norm_realdata.hdf5
4. get_pick_catalog.py
    1. Downloads a catalog of events from a client for a specified time range and minimum magnitude, then writes a .txt event file with rows for each of these events which include the origin time, event ID (based on whatever the client’s was), magnitude, latitude, longitude, and depth of the origin. After this, it makes .pick files for each of the events which include the pick time, network, station, and channel of the instrument they were picked on. 
    2. Only requires that you know what time range and magnitude range you want for your query and where to save things
5. add_stalocs_to_picktimes.py
    1. For each event in the event catalog, this code loops through the .pick files (one .pick file per event). For rows in the files which are picks on the vertical (Z) component (implied P picks), the code uses a station inventory to get the location of the station and other station info. It does the same for implied S picks on the horizontal (E or N) components. 
    2. Ultimate output: The .pick files for each event are transformed into two .npy files for each event (one for P picks and one for S picks) with the .npys containing the event ID, the pick time, the station network, name, and channel, and the latitude, longitude, and elevation of the station.
    3. Requires the event catalog text file, whatever info is needed to make the station inventory you want (i.e., network(s), stations and channels (can be ‘*’), latitude and longitude bounds and radius), and all of the .pick files for the events in the list
6. get_travel_times_from_picks.py
    1. Uses the .npys with P and S wave picks for each event (created in #4) and the .txt event catalog to subtract the pick time from the origin time and determine the travel times. The point of this is to have a simple integer to use for the interpolation to GNSS station locations.
    2. Ultimately output: New .npys for each event (one P and one S wave one) that have the event ID, the station name, latitude, and longitude, and the travel times for the waves to those stations from the origin.
    3. Requires the .npys from #4 and the event .txt file
7. interp_picks_to_GNSS_TUN.py
    1. For each event in the event catalog, this code loads the P and S wave travel time .npys from #5 and uses LinearNDInterpolator from SciPy to make a dense mesh map of the travel times between all of the stations (and saves figures of these maps). It then loops through the list of GNSS stations that you’re using for the CNN data and gets the interpolated travel time based on the closest point in the mesh map to the location of the GNSS station. 
    2. Ultimate output: Two new .npys for each event (one for the P waves and one for the S waves) that has the event ID, the GNSS station name, and the interpolated travel time to that station.
    3. Requires the .npys from #5, the .txt event catalog, and the list of GNSS stations for the project
8. traveltimes_to_picktimes_REDO.py (REDO in More_RealData; non-redo in Pick_stuff folder, not just RealData)
    1. Calculates the P and S wave arrival times at each GNSS station for each event in the catalog from the origin times and the travel times from the .npys in #6 
    2. Ultimate output: One new .npy containing ALL events and ALL arrival times (each row has the event ID, magnitude, station name, P arrival time, and S arrival time. nans if no arrivals for a particular row). 
9. find_eqs_in_realdata_metadata.py (in More_RealData, also in GNSS-CNN folder, not RealData)
    1. Loops through the rows in the metadata array associated with the real data that goes into the CNN, and then within these loops it loops through the array of interpolated GNSS P-wave arrival times from #7 to figure out which rows in the metadata array should have P-wave arrivals (and where by index)
    2. Ultimate output: A new metadata array that contains the same info as the one made in #2, but now each row has a new column that contains the index that the first P wave in the 128 second sample arrives at
        1. Named real_metadata_w_gauss_pos.npy
        2. Saves an array with the indices that have earthquakes (rowsweqs.npy) and the magnitudes of those earthquakes (rwe_mags.npy)
10. calc_PGD_SNR_cleaned.py (one in More_RealData is fixed)
    1. Uses the real data array to calculate the PGD of each 128s sample from the three components (normalizing them in the process), as well as the signal to noise ratio (20 seconds before and 20 seconds after the P-wave) for each of the three components. 
    2. Output is the average PGD and SNR (just printed) of the rows which the CNN got right in testing the real data versus the entire set of real earthquakes that should have been found
        1. Uses rowsweqs.npy for this from #8 as well as handpicked text file with rows that the CNN found earthquakes in ('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/newtrain_march/more_realdata_norm_testing/correct_indices.txt’)
    3. Builds a new metadata array that has all the info of #8 plus the PGD and SNR for the three components for the rows that do have earthquakes (nans otherwise)
        1. Named real_metadata_w_gauss_pgd_snr.npy in More_RealData

For implementation in CNN and analysis of tests

1. unet_3comp_training_logfeatures.py
    1. This code loads the hdf5 data and noise files, generates the data for the training, and trains the CNN
    2. It also contains the classification tests/figures for accuracy, etc. with the thresholds
    3. Also contains peak position tests/figures
    4. Also allows for running of testing data, including the real data
    5. Also saves the testing_for_analysis.npy array containing the metadata for the ruptures and adds the “result” column telling true pos, true neg, etc.
    6. Requires:
        1. FakeQuakes training data (hdf5)
        2. Noise data (ndf5)
        3. Metadata for the FakeQuakes (npy)
        4. Real data (hd5f) for testing
        5. Metadata for the real data (npy from #8 or #9 in previous section)
        6. gnss_unet_tools.py module code for the data generator
    7. Version in gnss-picker folder for GitHub also now saves the noisy fakequakes waveforms from the data generator (orig data) and the target gaussians (target) as origdata_fakequakes_testing.npy' and ‘target_fakequakes_testing.npy'

fqdata_analysiscodes

1. make_analysis_w_hypdist_csv.py (for FQ testing data)
    1. Loads the testing_for_analysis.npy array
    2. Writes analysis_w_hypdist_result.csv file that contains the event metadata (rupture, station, magnitude, pgd), the results, locations of earthquakes and stations, and the hypocentral distance. This code makes the CSV that DOES have the nans for noise events in it.
    3. Other option - this code, but getting PGD from the waveforms themselves rather than the log files. In the shuffle_trainfull folder
        1. Named make_analysis_w_hypdist_csv_VAL_pgdfromwaveforms.py 
2. calc_PGD_SNR_cleaned_fq.py (for FQ testing data)
    1. Calculates the PGDs and SNRs of the waveforms in the fakequakes testing data
    2. Uses origdata and target npy files from #7 in the unet code above 
    3. Produces 'SNRs_N_fakequakes_testing.npy' (or E or Z) for each of the three components
3. binning_SNREs_fqtesting.py (and SNRN and SNRZ)
    1. Puts the SNRs for each component into bins and then calculates the accuracy of the model’s predictions for each bin to make the combined accuracy plot
    2. Produces ‘E_barpos.txt’ (bar_positions) and ‘E_accper.txt' (accuracy_percentages) text files with each of the component’s codes to use for combined accuracy plot
4. binned_logSNRacc_allcomps.py (for FQ)
    1. Makes three component bar accuracy plot with Kraken colors. Uses text files from #4 
    2. Saves as binned_logacc_allcomps.png (need to fix and add SNR to this)
5. binning_mags.py (for FQ)
    1. Requires analysis_w_hypdist_result.csv file
    2. Bins magnitudes, calculates accuracy, and saves binned_mag_acc.png
6. binning_log_pgd.py (for FQ)
    1. Requires analysis_w_hypdist_result_waveformpgd.csv file
    2. Bins log PGDs, calculates accuracy, and saves binned_logwaveformpdg_acc.png

realdata_analysiscodes

1. test_real_classify.py 
    1. Code needs to be rewritten for clarity. Takes the second derivative of the waveforms and then checks to see how many zeros are in that, indicating that there was a data gap there that was interpolated across (making a straight line you wouldn’t expect a model to see an earthquake in, because there was no data there)
    2. Saves ‘toomanyzeros_N.npy’ arrays for each component which contain the indices where waveforms have too many zeros 
    3. Reads back in the ‘toomanyzeros_N.npy’ arrays, removes duplicates (since the three components could have the same numbers), and saves this list of indices as 'remove_dups_realdata.npy'
    4. Deletes the bad indices from the arrays for the waveforms, targets, predictions, and list of rows with earthquakes, saving these as:
        1. 'stack_data_rembad.npy' - waveforms
        2. 'realtest_predictions_rembad.npy' - model predictions
        3. 'gauss_target_rembad.npy' - targets
        4. 'rows_w_eqs_rembad.npy' - rows with earthquakes
    5. Loads the data back in that has bad samples removed, and does the classification tests to save text files with accuracies, precisions, etc. for the real data and makes the realdata_classifystats figures
2. realeq_correctorin_bymag.py
    1. Requires manually picked ‘correct_indices.txt’ file. The one currently in there is from newtrain_march while other things seem to be shuffle_trainfull? Need to redo
    2. Also requires the ‘rows_w_eqs_rembad.npy’ and metadata_gauss_pgd_snr file
    3. Produces the black and blue picked vs missed earthquakes by different properties plots




