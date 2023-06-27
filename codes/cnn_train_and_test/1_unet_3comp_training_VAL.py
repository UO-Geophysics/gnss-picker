#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:27:07 2020

Train a CNN to pick P and S wave arrivals with log features

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
from scipy import signal
import gnss_unet_tools
import argparse
import seaborn as sns

sns.set_style('white')

# # parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-subset', '--subset', help = 'train on a subset or no?', type = int)
# parser.add_argument('-pors', '--pors', help = 'train P or S network', type = int)
# parser.add_argument('-train', '--train', help = 'do you want to train?', type = int)
# parser.add_argument('-drop', '--drop', help = 'want to add a drop layer to the network', type = int)
# parser.add_argument('-plots', '--plots', help = 'want plots', type = int)
# parser.add_argument('-resume', '--resume', help = 'want to resume training?', type = int)
# parser.add_argument('-large', '--large', help = 'what size do you want the network to be?', type = float)
# parser.add_argument('-epochs', '--epochs', help = 'how many epochs', type = int)
# parser.add_argument('-std', '--std', help = 'standard deviation of target', type = float)
# parser.add_argument('-sr', '--sr', help = 'sample rate in hz', type = int)
# args = parser.parse_args()

##### VERSION NAME for figure saving #####

# name = 'new_train'
# name = '10_22_21' # Training with old distribution of data
name = 'shuffle_trainfull'
# local_dir = '/home/sdybing/GNSS-CNN_repo/GNSS-CNN/' # used for saving figures w/ Amanda's version

train = 0 # Do you want to train?
drop = 1 # Drop?
resume = 0 # Resume training
large = 0.5 # Large unet
epos = 100 # How many epochs?
std = 3 # How long do you want the Gaussian STD to be?
sr = 1 # Sample rate (Hz)

epsilon = 1e-6

print('train ' + str(train))
print('drop ' + str(drop))
print('resume ' + str(resume))
print('large ' + str(large))
print('epos ' + str(epos))
print('std ' + str(std))
print('sr ' + str(sr))

##### -------------------- LOAD THE DATA -------------------- #####

print('LOADING DATA')

# Signals of earthquakes
x_data = h5py.File('nd3_data.hdf5', 'r')
x_data = x_data['nd3_data'][:,:]
# print(x_data.shape)

# Noise data
n_data = h5py.File('729k_noise.hdf5', 'r')
n_data = n_data['729k_noise'][:,:]

# Metadata with information about earthquakes in x_data
meta_data = np.load('nd3_info.npy')
# shuffle_meta = meta_data[:] # Initialize for shuffling later
# print('Meta shape:')
# print(meta_data.shape)

# Array of NaNs to use to match added noise in concatenation later
nan_array = np.empty((len(x_data), 3))
nan_array[:] = np.NaN

## Real data things

real_data = h5py.File('realdata_data.hdf5', 'r')
real_data = real_data['realdata_data'][:,:] # shape: (12240, 384)
real_meta_data = np.load('real_metadata_w_gauss_pgd_snr.npy') # shape: (12240, 6). Station name, date, sample start time, end time, counter, and position for Gaussian peak if there should be one
real_data_inds = np.arange(real_data.shape[0])

## New big set real data things (normed)

norm_real_data = h5py.File('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/norm_realdata.hdf5', 'r')
norm_real_data = norm_real_data['norm_realdata'][:,:]
norm_real_data_inds = np.arange(norm_real_data.shape[0])
norm_real_meta_data = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pgd_snr.npy')

# model_save_file = 'gnssunet_3comp_logfeat_250000_pn_eps_' + str(epos) + '_sr_' + str(sr) + '_std_' + str(std) + '.tf' 
model_save_file = 'gnssunet_' + name + '.tf' 
        
if large:
    fac = large
    model_save_file = 'large_' + str(fac) + '_' + model_save_file

if drop:
    model_save_file = 'drop_' + model_save_file

##### -------------------- MAKE TRAINING AND TESTING DATA -------------------- #####

print('MAKE TRAINING AND TESTING DATA')

np.random.seed(0)
# print(np.random.seed(0))

# # Training

# # Shuffling indices of earthquake data and grabbing 90%, then putting them back in order
# siginds = np.arange(x_data.shape[0]) # numbers between 0 and 100,043
# np.random.shuffle(siginds) # randomly shuffles the numbers between 0 and 100,043
# sig_train_inds = np.sort(siginds[:int(0.9*len(siginds))]) # grabs the front 90% of the numbers, then sorts them back into order

# # Shuffling indices of noise data and grabbing 90%, then putting them back in order
# noiseinds = np.arange(n_data.shape[0])
# np.random.shuffle(noiseinds)
# noise_train_inds = np.sort(noiseinds[:int(0.9*len(noiseinds))]) # try getting rid of sort?

# ## Testing
# sig_test_inds = np.sort(siginds[int(0.9*len(siginds)):]) # grabs the back 10% (90% through the end) and sorts 
# noise_test_inds = np.sort(noiseinds[int(0.9*len(noiseinds)):])

# Trying now shuffling without resorting to see if it fixes the distribution issue. Also shuffling metadata

# Shuffling indices of earthquake data and grabbing 90%, then putting them back in order
siginds = np.arange(x_data.shape[0]) # numbers between 0 and 100,043
print(siginds)
np.random.shuffle(siginds) # randomly shuffles the numbers between 0 and 100,043
print(siginds)
print(siginds.shape)
sig_train_inds = siginds[:int(0.9*len(siginds))] # grabs the front 90% of the numbers, then sorts them back into order
print(sig_train_inds)
print(sig_train_inds.shape)

# Shuffling indices of noise data and grabbing 90%, then putting them back in order
noiseinds = np.arange(n_data.shape[0])
np.random.shuffle(noiseinds)
noise_train_inds = noiseinds[:int(0.9*len(noiseinds))] # try getting rid of sort?

# Shuffling metadata
# np.random.shuffle(shuffle_meta)
# print(shuffle_meta)
# print('Shuffle meta shape:')
# print(shuffle_meta.shape)
# Replacing meta_data with shuffle_meta later in code 

## Testing
sig_test_inds = siginds[int(0.9*len(siginds)):] # grabs the back 10% (90% through the end) and sorts 
noise_test_inds = noiseinds[int(0.9*len(noiseinds)):]

# Some plots to check what we've loaded
    
# Plot the data (no noise, just earthquakes)
plt.figure(figsize = (8,5))   
for ii in range(10): # plot 20 of them
    plt.plot(x_data[ii,:]/np.max(np.abs(x_data[ii,:]))+ii) # normalized
plt.title('Earthquake test')
plt.xlabel('Time (s)')
plt.ylabel('Normalized amplitude')
# plt.savefig(local_dir + 'plots/' + name + '/1_plot_raw_eq_data.png', format = 'PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/1_2_raw_data_noise/plot_raw_eq_data.png', format = 'PNG') # LAP
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/1_2_raw_data_noise/plot_raw_eq_data.png', format = 'PNG') # VAL

plt.close()

# Plot noise to check
plt.figure(figsize = (8,5))
for ii in range(10):
    plt.plot(n_data[ii,:]/np.max(np.abs(n_data[ii,:]))+ii) # normalized?
plt.title('Noise test')
plt.xlabel('Time (s)')
plt.ylabel('Normalized amplitude')
# plt.savefig(local_dir + 'plots/' + name + '/2_plot_noise_data.png', format = 'PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/1_2_raw_data_noise/plot_noise_data.png', format = 'PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/1_2_raw_data_noise/plot_noise_data.png', format = 'PNG')
plt.close()

### Check the random PGD distribution ###

testing_data = x_data[sig_test_inds]
print(len(testing_data))

pgd=np.zeros(testing_data.shape[0])
for ii in range(testing_data.shape[0]):
    pgd[ii]=np.max(np.sqrt((testing_data[ii,:256])**2+(testing_data[ii,256:512])**2+(testing_data[ii,512:])**2))

plt.figure(figsize=(8,5))
plt.hist(pgd,bins=np.arange(0,5,0.05), color=(162/256,210/256,255/256),alpha=0.5, edgecolor='black', linewidth=1.2)
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/1_2_raw_data_noise/test_pgd_distrib.png', format = 'PNG') # LAP
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/1_2_raw_data_noise/test_pgd_distrib.png', format = 'PNG') # LAP
plt.close()

# ##### -------------------- FIRST GENERATOR TEST: generate batch data -------------------- #####

print('FIRST PASS WITH DATA GENERATOR')

my_data = gnss_unet_tools.my_3comp_data_generator(32, x_data, n_data, meta_data, nan_array, sig_train_inds, noise_train_inds, sr, std, valid = True) # Valid = True to get original data back
batch_out, target, origdata, metadata = next(my_data) # batch_out and origdata are the same with GNSS implementation
# Shapes:
    # batch_out: (batch_size, 128, 3) # N, E, Z
    # target: (5000, 128)
    # origdata: (5000, 128, 3) # N, E, Z
    # metadata: (5000, 3) Rupt name, station name, magnitude

# Plot generator results

nexamples = 5 # Number of examples to look at 
  
for ind in range(nexamples): 
    
    # fig = plt.subplots(nrows = 1, ncols = 3, figsize = (18,4), dpi = 300)
    fig = plt.subplots(nrows = 1, ncols = 3, figsize = (26,4), dpi = 300) # shoter for AGU talk
    plt.subplots_adjust(wspace = 0.4)
    t = 1/sr * np.arange(batch_out.shape[1])
    
    ax1 = plt.subplot(131)
    ax1.plot(t, origdata[ind,:,0]*100, label = 'N original data', color = 'C0')
    ax1.set_ylabel('Displacement (cm)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc = 'upper right')
    ax2 = ax1.twinx()
    ax2.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax2.set_ylabel('Confidence')
    ax2.legend(loc = 'lower right')
    
    ax3 = plt.subplot(132)
    ax3.plot(t, origdata[ind,:,1]*100, label = 'E original data', color = 'C1')
    ax3.set_ylabel('Displacement (cm)')
    ax3.legend(loc = 'upper right')
    ax4 = ax3.twinx()
    ax4.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax4.legend(loc = 'lower right')
    
    ax5 = plt.subplot(133)
    ax5.plot(t, origdata[ind,:,2]*100, label = 'Z original data', color = 'C2')
    ax5.set_ylabel('Displacement (cm)')
    ax5.legend(loc = 'upper right')
    ax6 = ax5.twinx()
    ax6.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax6.legend(loc = 'lower right')
    
    # plt.savefig(local_dir + 'plots/' + name + '/3_ex_' + str(ind) + '_plot_generator_pass.png', format = 'PNG')
    plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/3_first_pass_gen/ex_' + str(ind) + '_plot_generator_pass.png', format = 'PNG')
    # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/3_first_pass_gen/ex_' + str(ind) + '_plot_generator_pass.png', format = 'PNG')
    plt.close()
    

# ##### -------------------- BUILD THE MODEL -------------------- #####

print('BUILD THE MODEL')

if drop:
    model = gnss_unet_tools.make_large_unet_drop(fac, sr, ncomps = 3)
    print('Using drop model')    
    
else:
    model = gnss_unet_tools.make_large_unet(fac, sr, ncomps = 3)  
    print('Using large model')
        
# ##### -------------------- ADD SOME CHECKPOINTS -------------------- #####

print('ADDING CHECKPOINTS')

checkpoint_filepath = './checks/' + model_save_file + '_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath, save_weights_only = True, verbose = 1,
    monitor = 'val_accuracy', mode = 'max', save_best_only = True) # rename val_loss to validation_loss, or val_acc to val_accuracy

# ##### -------------------- TRAIN THE MODEL -------------------- #####

if train:
    
    print('TRAINING!!!')
    
    batch_size = 32 # Why 32?
    
    if resume:
        print('Resuming training results from ' + model_save_file)
        model.load_weights(checkpoint_filepath)
        
    else:
        print('Training model and saving results to ' + model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file + '.csv', append = True)
    
    history = model.fit_generator(gnss_unet_tools.my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_train_inds, noise_train_inds, sr, std), # Valid = False for training; implied
                        steps_per_epoch = (len(sig_train_inds) + len(noise_train_inds))//batch_size,
                        validation_data = gnss_unet_tools.my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_test_inds, noise_test_inds, sr, std),
                        validation_steps = (len(sig_test_inds) + len(noise_test_inds))//batch_size,
                        epochs = epos, callbacks = [model_checkpoint_callback, csv_logger])
    
    model.save_weights('./' + model_save_file)
    
else:
    print('LOADING TRAINING RESULTS from ' + model_save_file)
    model.load_weights('./' + model_save_file)
    
# Plotting training curves

print('RUNNING ANALYSES')

training_stats = np.genfromtxt('./' + model_save_file + '.csv', delimiter = ',', skip_header = 1)

fig = plt.subplots(nrows = 2, ncols = 1, figsize = (6,8))
plt.suptitle('Training curves')

ax1 = plt.subplot(211)
ax1.plot(training_stats[1:100,0], training_stats[1:100,1], label = 'Accuracy')
ax1.plot(training_stats[1:100,0], training_stats[1:100,3], label = 'Validation accuracy') 
ax1.legend(loc = 'lower right')
ax1.set_title(model_save_file)

ax2 = plt.subplot(212)
ax2.plot(training_stats[1:100,0], training_stats[1:100,2], label = 'Loss') 
ax2.plot(training_stats[1:100,0], training_stats[1:100,4], label = 'Validation loss') 
ax2.legend(loc = 'upper right')
ax2.set_xlabel('Epoch')

# plt.savefig(local_dir + 'plots/' + name + '/4_training_curves.png', format = 'PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/4_training_curves/training_curves.png', format = 'PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/4_training_curves/training_curves.png', format = 'PNG')
plt.close()

##### -------------------- TEST THE MODEL -------------------- #####

print('TESTING!!!')

# Number of samples to test with
num_test = len(testing_data) - 1
# print(num_test)

# See how things went with the remaining 10%
my_test_data = gnss_unet_tools.my_3comp_data_generator(num_test, x_data, n_data, meta_data, nan_array, sig_test_inds, noise_test_inds, sr, std, valid = True)
batch_out, target, origdata, metadata = next(my_test_data)
test_predictions = model.predict(batch_out)

# print('test_predictions shape:')
# print(test_predictions.shape)
# print('target shape:')
# print(target.shape)
# print(target[1])
# print(batch_out.shape)
# print(origdata.shape)

# Calculate PGDs and distribution

pgd_1=np.zeros(origdata.shape[0])
for ii in range(origdata.shape[0]):
    pgd_1[ii]=np.max(np.sqrt((origdata[ii,:,0])**2+(origdata[ii,:,1])**2+(origdata[ii,:,2])**2))

plt.figure(figsize=(8,5), dpi=300)
plt.hist(pgd_1,bins=np.arange(0,5,0.05), color=(162/256,210/256,255/256),alpha=0.5, edgecolor='black', linewidth=1.2)
plt.ylim(0,4000)
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/5_10per_testing/pgd_distrib.png', format = 'PNG') # LAP
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/5_10per_testing/pgd_distrib.png', format = 'PNG') # LAP
plt.close()

np.save('origdata_fakequakes_testing.npy', origdata)
np.save('target_fakequakes_testing.npy', target)

# Plot some data and predictions!

nexamples = 20 # Number of examples to look at 
  
for ind in range(nexamples): 
    
    fig = plt.subplots(nrows = 1, ncols = 3, figsize = (18,4), dpi = 300)
    plt.subplots_adjust(wspace = 0.4)
    t = 1/sr * np.arange(batch_out.shape[1])
    # print(t)
    
    ax1 = plt.subplot(131)
    ax1.plot(t, origdata[ind,:,0]*100, label = 'N original data', color = 'C0')
    ax1.set_ylabel('Displacement (cm)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc = 'upper right')
    ax2 = ax1.twinx()
    ax2.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax2.plot(t, test_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(-0.05,1.05)
    ax2.legend(loc = 'upper left')
    
    ax3 = plt.subplot(132)
    ax3.plot(t, origdata[ind,:,1]*100, label = 'E original data', color = 'C1')
    ax3.set_ylabel('Displacement (cm)')
    ax3.legend(loc = 'upper right')
    ax4 = ax3.twinx()
    ax4.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax4.plot(t, test_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
    ax4.set_ylim(-0.05,1.05)
    ax4.legend(loc = 'upper left')
    
    ax5 = plt.subplot(133)
    ax5.plot(t, origdata[ind,:,2]*100, label = 'Z original data', color = 'C2')
    ax5.set_ylabel('Displacement (cm)')
    ax5.legend(loc = 'upper right')
    ax6 = ax5.twinx()
    ax6.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
    ax6.plot(t, test_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
    ax6.set_ylim(-0.05,1.05)
    ax6.legend(loc = 'upper left')
    
    # plt.savefig(local_dir + 'plots/' + name + '/5_ex_' + str(ind) + '_plot_predictions.png', format = 'PNG')
    plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/5_10per_testing/ex_' + str(ind) + '_plot_predictions.png', format = 'PNG')
    # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/5_10per_testing/ex_' + str(ind) + '_plot_predictions.png', format = 'PNG')
    plt.close()

# ##### -------------------- REAL DATA TESTS -------------------- #####

# print('TESTING WITH REAL DATA!!!')

# # realtest = gnss_unet_tools.real_data_generator(data = real_data, data_inds = real_data_inds, meta_data = real_meta_data, sr = 1, std = 3, nlen = 128)
# realtest = gnss_unet_tools.real_data_generator(data = norm_real_data, data_inds = norm_real_data_inds, meta_data = norm_real_meta_data, sr = 1, std = 3, nlen = 128)
# stack_data, gauss_target = next(realtest)
# # print(stack_data)
# # print(stack_data.shape)
# # print(gauss_target)
# # print(gauss_target.shape)
# realtest_predictions = model.predict(stack_data)

# np.save('realtest_predictions.npy', realtest_predictions)
# np.save('stack_data.npy', stack_data)
# np.save('gauss_target.npy', gauss_target)

# # Plot some data and predictions!

# # rows_w_eqs = np.load('real_metadata_rowsweqs.npy')
# rows_w_eqs = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rowsweqs.npy') # More data, normed

# nexamples = 5 # Number of examples to look at 
  
# # for ind in range(nexamples): 
# for ind in rows_w_eqs[43]:
    
#     fig = plt.subplots(nrows = 1, ncols = 3, figsize = (22,4), dpi = 300)
#     plt.subplots_adjust(wspace = 0.4)
#     t = 1/sr * np.arange(batch_out.shape[1])
    
#     ax1 = plt.subplot(131)
#     ax1.plot(t, stack_data[ind,:,0]*100, label = 'N original data', color = 'C0')
#     ax1.set_ylabel('Displacement (cm)')
#     ax1.set_xlabel('Time (s)')
#     ax1.legend(loc = 'upper right')
#     ax2 = ax1.twinx()
#     ax2.plot(t, gauss_target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#     ax2.plot(t, realtest_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
#     ax2.set_ylabel('Confidence')
#     ax2.set_ylim(-0.05,1.05)
#     ax2.legend(loc = 'upper left')
    
#     ax3 = plt.subplot(132)
#     ax3.plot(t, stack_data[ind,:,1]*100, label = 'E original data', color = 'C1')
#     ax3.set_ylabel('Displacement (cm)')
#     ax3.legend(loc = 'upper right')
#     ax4 = ax3.twinx()
#     ax4.plot(t, gauss_target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#     ax4.plot(t, realtest_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
#     ax4.set_ylim(-0.05,1.05)
#     ax4.legend(loc = 'upper left')
    
#     ax5 = plt.subplot(133)
#     ax5.plot(t, stack_data[ind,:,2]*100, label = 'Z original data', color = 'C2')
#     ax5.set_ylabel('Displacement (cm)')
#     ax5.legend(loc = 'upper right')
#     ax6 = ax5.twinx()
#     ax6.plot(t, gauss_target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#     ax6.plot(t, realtest_predictions[ind,:], color = 'red', linestyle = '--', label = 'Prediction')
#     ax6.set_ylim(-0.05,1.05)
#     ax6.legend(loc = 'upper left')
    
#     # plt.show()
    
#     # plt.savefig(local_dir + 'plots/' + name + '/5_ex_' + str(ind) + '_plot_predictions.png', format = 'PNG')
#     plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/6_real_testing/ex_' + str(ind) + '_plot_predictions.png', format = 'PNG') # First small real test
#     # plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/newtrain_march/more_realdata_norm_testing/ex_' + str(ind) + '_plot_preds.png', format = 'PNG') # Big real set, normed
#     # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/6_real_testing/ex_' + str(ind) + '_plot_predictions.png', format = 'PNG')
#     plt.close()

        
##### -------------------- CLASSIFICATION TESTS -------------------- #####

# print('DOING CLASSIFICATION TESTS ON FAKEQUAKES DATA')

# # Decision threshold evaluation

# thresholds = np.arange(0, 1.005, 0.005)
# # thresholds = np.arange(0, 1, 0.1)
# test_thresholds = [0]

# # Use np.where to see whether anywhere in test_predictions is > threshold
# # If there is a value that's >, the 'result' of the array is 1. If not 0
# # Then compare these 1s and 0s to the target array value for PAR

# accuracies = []
# accuracies_per = []
# precisions = []
# recalls = []
# F1s = []

# for threshold in thresholds:
    
#     # print('-------------------------------------------------------------')
#     # print('Threshold: ' + str(threshold))
#     # print('-------------------------------------------------------------')
#     # print(' ')
    
#     # Convert the predictions arrays to single ones and zeroes
    
#     pred_binary = np.zeros(len(test_predictions))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         # print('Prediction: ' + str(test_predictions[k]))
#         i = np.where(test_predictions[k] >= threshold)[0]
#         # print(i)
#         if len(i) == 0:
#             pred_binary[k] = 0
#         elif len(i) > 0:
#             pred_binary[k] = 1
    
#     # print('Predictions: ')
#     # print(pred_binary)
#     # print(pred_binary.shape)
    
#     # Convert the target arrays to single ones and zeroes
    
#     targ_binary = np.zeros(len(target))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         # print('Target: ' + str(test_predictions[k]))
#         i = np.where(target[k] > 0)[0]
#         if len(i) == 0:
#             targ_binary[k] = 0
#         elif len(i) > 0:
#             targ_binary[k] = 1
    
#     # print('Targets: ')
#     # print(targ_binary)
    
#     # Calculating the accuracy, precision, recall, and F1
    
#     num_preds = num_test # total number of predictions. Amanda did 50
#     correct_preds = []
#     wrong_preds = []
#     true_pos = []
#     true_neg = []
#     false_pos = []
#     false_neg = []
    
#     for i in iterate:
        
#         pred = pred_binary[i]
#         targ = targ_binary[i]
        
#         if pred == targ: # add one to list of correct predictions if matching
#             correct_preds.append(1)
            
#             if pred == 1 and targ == 1:
#                 true_pos.append(1)
#             elif pred == 0 and targ == 0:
#                 true_neg.append(1)
            
#         elif pred != targ: # add ones to list of incorrect predictions if not matching
#             wrong_preds.append(1)
            
#             if pred == 1 and targ == 0:
#                 false_pos.append(1)
#             elif pred == 0 and targ == 1:
#                 false_neg.append(1)
    
#     num_correct_preds = len(correct_preds)
#     num_wrong_preds = len(wrong_preds)
#     num_true_pos = len(true_pos)
#     num_true_neg = len(true_neg)
#     num_false_pos = len(false_pos)
#     num_false_neg = len(false_neg)
    
#     # print('Correct preds: ' + str(num_correct_preds))
#     # print('Wrong preds: ' + str(num_wrong_preds))
#     # print('True pos: ' + str(num_true_pos))
#     # print('True neg: ' + str(num_true_neg))
#     # print('False pos: ' + str(num_false_pos))
#     # print('False neg: ' + str(num_false_neg))
    
#     # print('Threshold: ' + str(threshold))
#     # print('Correct preds: ' + str(num_correct_preds))
#     # print('Wrong preds: ' + str(num_wrong_preds))
#     # print('True pos: ' + str(num_true_pos))
#     # print('True neg: ' + str(num_true_neg))
#     # print('False pos: ' + str(num_false_pos))
#     # print('False neg: ' + str(num_false_neg))
    
#     accuracy = num_correct_preds / num_preds
#     accuracy_per = (num_correct_preds / num_preds) * 100
#     # print('Accuracy: ' + str(accuracy_per) + '%')
    
#     if num_true_pos == 0  and num_false_pos == 0:
#         precision = 0
#     else:
#         precision = num_true_pos / (num_true_pos + num_false_pos)
    
#     if num_true_pos == 0 and num_false_neg == 0:
#         recall = 0
#     else:
#         recall = num_true_pos / (num_true_pos + num_false_neg)
    
#     if precision + recall == 0:
#         F1 = 0
#     else:
#         F1 = 2 * ((precision * recall) / (precision + recall))
    
#     accuracies.append(accuracy)
#     accuracies_per.append(accuracy_per)
#     precisions.append(precision)
#     recalls.append(recall)
#     F1s.append(F1)

# # print('Accuracies')
# # print(accuracies)
# # print('Precisions')
# # print(precisions)
# # print('Recalls')
# # print(recalls)
# # print('F1s')
# # print(F1s)

# np.savetxt('accuracies_percentage_txt.txt', accuracies_per)
# np.savetxt('thresholds_txt.txt', thresholds)

# plt.figure(figsize = (8,5), dpi = 300)
# # plt.scatter(thresholds,accuracies)
# plt.plot(thresholds, accuracies_per, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Accuracy (%)', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,100)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Accuracy Percentage', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_accuracies_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/Users/sydneydybing/Documents/AGU_2021/Figures/6_accuracies_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/7_classify_stats/accuracies_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/accuracies_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, precisions, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Precision', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Precision', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_precisions_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/7_classify_stats/precisions_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/precisions_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, recalls, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Recall', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Recall', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_recalls_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/7_classify_stats/recalls_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/recalls_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, F1s, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('F1', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('F1', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_F1s_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/7_classify_stats/F1s_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/F1s_' + str(num_test) + '.png', format='PNG')
# plt.close()

# ##### -------------------- GAUSSIAN PEAK POSITION TEST -------------------- #####

# print('DOING PEAK POSITION TESTS')

# # print(target[4])
# # print(test_predictions[4])

# thresholds = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])

# # threshold = 0.2

# iterate = np.arange(0,num_test,1)
# s = 0

# fig = plt.subplots(nrows = 3, ncols = 4, figsize = (18,14), dpi = 400)
# # fig = plt.subplots(nrows = 3, ncols = 4, figsize = (18,14))
# plt.suptitle('Target vs. Prediction Samples Off per Threshold', fontsize = 20)

# for idx in range(len(thresholds)):
    
#     threshold = thresholds[idx]

#     pred_binary = np.zeros(len(test_predictions))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(test_predictions[k] >= threshold)[0]
#         if len(i) == 0:
#             pred_binary[k] = 0
#         elif len(i) > 0:
#             pred_binary[k] = 1
    
#     # print('Predictions: ')
#     # print(pred_binary)
    
#     # Convert the target arrays to single ones and zeroes
    
#     targ_binary = np.zeros(len(target))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(target[k] > 0)[0]
#         if len(i) == 0:
#             targ_binary[k] = 0
#         elif len(i) > 0:
#             targ_binary[k] = 1
    
#     # print('Targets: ')
#     # print(targ_binary)
    
#     signals = []
    
#     for i in iterate:
#         pred = pred_binary[i]
#         targ = targ_binary[i]
        
#         # print(pred)
#         # print(targ)
        
#         if pred == 1 and targ == 1: # True positive, there was a signal and it found it
#             signals.append(i) # Grab index from list of events that are correct and have a pick
#         else:
#             pass
    
#     # print(signals)
    
#     samples_off_list = []
    
#     for index in signals:
        
#         # Find the peak and then the index where that peak is and compare 
        
#         # print('----------------------')
#         # print('Signal number: ' + str(index))
        
#         target_max_idx = np.argmax(target[index])
#         # print('Target: ' + str(target_max_idx))
        
#         pred_max_idx = np.argmax(test_predictions[index])
#         # print('Prediction: ' + str(pred_max_idx))
        
#         samples_off = np.abs(pred_max_idx - target_max_idx)
#         # print('Samples off: ' + str(samples_off))
#         samples_off_list.append(samples_off)
        
#     # print(samples_off_list)
    
#     plt.subplot(3,4,idx+1)
#     plt.hist(samples_off_list, bins=128, range=(0,128), label = 'Threshold: ' + str(threshold))
#     plt.xlim(0,128)
#     plt.ylim(0,300)
#     plt.legend()
#     plt.grid(which = 'major', color = 'lightgray')
#     plt.subplots_adjust(hspace = 0, wspace = 0)

#     if idx == 0:
#         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
#     elif idx == 4:
#         plt.ylabel('Number of examples in bin')
#         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
#         plt.yticks([0, 50, 100, 150, 200, 250])
        
#     elif idx == 8:
#         plt.yticks([0, 50, 100, 150, 200, 250])
        
#     elif idx == 9:
#         plt.xlabel('Numbers of samples off target position')
#         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
        
#     elif idx == 10:     
#         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
        
#     else:
#         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
#         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
    
#     plt.subplot(3,4,12)
#     plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
#     plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)

# # plt.savefig(local_dir + 'plots/' + name + '/7_histogram.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/8_peakpos_stats/histogram.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/8_peakpos_stats/histogram.png', format='PNG')
# plt.close()

# ##### -------------------- METADATA ANALYSIS -------------------- #####

# print('DOING METADATA ANALYSIS')

# print(batch_out.shape) # the data
# print(metadata.shape) # the metadata
# print(test_predictions.shape) # the model's predictions about the data

# # np.save('batch_out_11_10_21.npy', batch_out)  
# # np.save('test_preds_11_10_21.npy', test_predictions)  
# # np.save('target_11_10_21.npy', target)

# # print(batch_out[0])
# # print(metadata[0][0])
# # print(test_predictions[0])

# # print(metadata)

# zeros = np.zeros((test_predictions.shape[0],1))
# analysis_array = np.c_[metadata,zeros]
# # print(analysis_array.shape)

# for i in range(len(batch_out)):
    
#     # print(i)
    
#     # print(metadata[i])

#     if metadata[i][0] == 'nan':
#         # print(str(i) + ' is not an earthquake')
#         # analysis_array[i][3] = 'nan'
        
#         threshold = 0.615
        
#         # True positive, true negative, false positive, false negative
        
#         # print('Threshold: ' + str(threshold))
    
#         # Convert the predictions arrays to single ones and zeroes
        
#         p = np.where(test_predictions[i] >= threshold)[0]
#         if len(p) == 0:
#             pred_binary = 0
#         elif len(p) > 0:
#             pred_binary = 1
        
#         # if i == 0:
#         #     print('Prediction: ')
#         #     print(pred_binary)
        
#         # # Convert the target arrays to single ones and zeroes
        
#         t = np.where(target[i] > 0)[0]
#         if len(t) == 0:
#             targ_binary = 0
#         elif len(t) > 0:
#             targ_binary = 1
        
#         # if i == 0:
#         #     print('Target: ')
#         #     print(targ_binary)
        
#         pred = pred_binary
#         targ = targ_binary
        
#         if pred == targ: # add one to list of correct predictions if matching
#             # correct_preds.append(1)
            
#             if pred == 1 and targ == 1:
#                 result = 'true pos'
#             elif pred == 0 and targ == 0:
#                 result = 'true neg'
            
#         elif pred != targ: # add ones to list of incorrect predictions if not matching
#             # wrong_preds.append(1)
            
#             if pred == 1 and targ == 0:
#                 result = 'false pos'
#             elif pred == 0 and targ == 1:
#                 result = 'false neg'
        
#         analysis_array[i][3] = result
    
#     else:
#         # print(str(i) + ' is an earthquake')
        
#         rupt_num = metadata[i][0]
#         station = metadata[i][1]
#         mag = metadata[i][2]
        
#         # print(rupt_num)
#         # print(station)
#         # print(mag)
        
#         # print(batch_out[i])
#         # print(test_predictions[i])
#         # plt.plot(test_predictions[i])
#         # plt.show()
        
#         threshold = 0.2
        
#         # True positive, true negative, false positive, false negative
        
#         # print('Threshold: ' + str(threshold))
    
#         # Convert the predictions arrays to single ones and zeroes
        
#         p = np.where(test_predictions[i] >= threshold)[0]
#         if len(p) == 0:
#             pred_binary = 0
#         elif len(p) > 0:
#             pred_binary = 1
        
#         # if i == 0:
#         #     print('Prediction: ')
#         #     print(pred_binary)
        
#         # # Convert the target arrays to single ones and zeroes
        
#         t = np.where(target[i] > 0)[0]
#         if len(t) == 0:
#             targ_binary = 0
#         elif len(t) > 0:
#             targ_binary = 1
        
#         # if i == 0:
#         #     print('Target: ')
#         #     print(targ_binary)
        
#         pred = pred_binary
#         targ = targ_binary
        
#         if pred == targ: # add one to list of correct predictions if matching
#             # correct_preds.append(1)
            
#             if pred == 1 and targ == 1:
#                 result = 'true pos'
#             elif pred == 0 and targ == 0:
#                 result = 'true neg'
            
#         elif pred != targ: # add ones to list of incorrect predictions if not matching
#             # wrong_preds.append(1)
            
#             if pred == 1 and targ == 0:
#                 result = 'false pos'
#             elif pred == 0 and targ == 1:
#                 result = 'false neg'
        
#         analysis_array[i][3] = result
    
# print(analysis_array)
# print(analysis_array.shape)
            
# # np.save('/home/sdybing/GNSS_project/' + name + 'testing_for_analysis.npy', analysis_array) # VAL
# np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/testing_for_analysis.npy', analysis_array) # LAP        

# # ##### -------------------- CLASSIFICATION TESTS -------------------- #####

# print('DOING CLASSIFICATION TESTS ON REAL DATA')

# # Decision threshold evaluation

# thresholds = np.arange(0, 1.005, 0.005)
# # thresholds = np.arange(0, 1, 0.1)
# test_thresholds = [0]

# # Use np.where to see whether anywhere in test_predictions is > threshold
# # If there is a value that's >, the 'result' of the array is 1. If not 0
# # Then compare these 1s and 0s to the target array value for PAR

# accuracies = []
# accuracies_per = []
# precisions = []
# recalls = []
# F1s = []

# for threshold in thresholds:
    
#     # print('-------------------------------------------------------------')
#     # print('Threshold: ' + str(threshold))
#     # print('-------------------------------------------------------------')
#     # print(' ')
    
#     # Convert the predictions arrays to single ones and zeroes
    
#     pred_binary = np.zeros(len(realtest_predictions))
#     iterate = np.arange(0,len(realtest_predictions),1)
    
#     for k in iterate:
#         # print('Prediction: ' + str(test_predictions[k]))
#         i = np.where(realtest_predictions[k] >= threshold)[0]
#         # print(i)
#         if len(i) == 0:
#             pred_binary[k] = 0
#         elif len(i) > 0:
#             pred_binary[k] = 1
    
#     # print('Predictions: ')
#     # print(pred_binary)
#     # print(pred_binary.shape)
    
#     # Convert the target arrays to single ones and zeroes
    
#     targ_binary = np.zeros(len(gauss_target)) # Need to make this ones at indices in rows_w_eqs
    
#     for idx in range(len(targ_binary)):
        
#         if idx in rows_w_eqs:
            
#             targ_binary[idx] = 1
    
#     # print('Targets: ')
#     # print(targ_binary)
    
#     # Calculating the accuracy, precision, recall, and F1
    
#     num_preds = len(realtest_predictions) # total number of predictions. Amanda did 50
#     correct_preds = []
#     wrong_preds = []
#     true_pos = []
#     true_neg = []
#     false_pos = []
#     false_neg = []
    
#     for i in iterate:
        
#         pred = pred_binary[i]
#         targ = targ_binary[i]
        
#         if pred == targ: # add one to list of correct predictions if matching
#             correct_preds.append(1)
            
#             if pred == 1 and targ == 1:
#                 true_pos.append(1)
#             elif pred == 0 and targ == 0:
#                 true_neg.append(1)
            
#         elif pred != targ: # add ones to list of incorrect predictions if not matching
#             wrong_preds.append(1)
            
#             if pred == 1 and targ == 0:
#                 false_pos.append(1)
#             elif pred == 0 and targ == 1:
#                 false_neg.append(1)
    
#     num_correct_preds = len(correct_preds)
#     num_wrong_preds = len(wrong_preds)
#     num_true_pos = len(true_pos)
#     num_true_neg = len(true_neg)
#     num_false_pos = len(false_pos)
#     num_false_neg = len(false_neg)
    
#     # print('Correct preds: ' + str(num_correct_preds))
#     # print('Wrong preds: ' + str(num_wrong_preds))
#     # print('True pos: ' + str(num_true_pos))
#     # print('True neg: ' + str(num_true_neg))
#     # print('False pos: ' + str(num_false_pos))
#     # print('False neg: ' + str(num_false_neg))
    
#     # print('Threshold: ' + str(threshold))
#     # print('Correct preds: ' + str(num_correct_preds))
#     # print('Wrong preds: ' + str(num_wrong_preds))
#     # print('True pos: ' + str(num_true_pos))
#     # print('True neg: ' + str(num_true_neg))
#     # print('False pos: ' + str(num_false_pos))
#     # print('False neg: ' + str(num_false_neg))
    
#     accuracy = num_correct_preds / num_preds
#     accuracy_per = (num_correct_preds / num_preds) * 100
#     # print('Accuracy: ' + str(accuracy_per) + '%')
    
#     if num_true_pos == 0  and num_false_pos == 0:
#         precision = 0
#     else:
#         precision = num_true_pos / (num_true_pos + num_false_pos)
    
#     if num_true_pos == 0 and num_false_neg == 0:
#         recall = 0
#     else:
#         recall = num_true_pos / (num_true_pos + num_false_neg)
    
#     if precision + recall == 0:
#         F1 = 0
#     else:
#         F1 = 2 * ((precision * recall) / (precision + recall))
    
#     accuracies.append(accuracy)
#     accuracies_per.append(accuracy_per)
#     precisions.append(precision)
#     recalls.append(recall)
#     F1s.append(F1)

# # print('Accuracies')
# # print(accuracies)
# # print('Precisions')
# # print(precisions)
# # print('Recalls')
# # print(recalls)
# # print('F1s')
# # print(F1s)

# np.savetxt('realdata_accuracies_percentage_txt.txt', accuracies_per)
# np.savetxt('realdata_thresholds_txt.txt', thresholds)

# plt.figure(figsize = (8,5), dpi = 300)
# # plt.scatter(thresholds,accuracies)
# plt.plot(thresholds, accuracies_per, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Accuracy (%)', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,100)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Accuracy Percentage', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_accuracies_' + str(num_test) + '.png', format='PNG')
# # plt.savefig('/Users/sydneydybing/Documents/AGU_2021/Figures/6_accuracies_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/accuracies_realdata.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/accuracies_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, precisions, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Precision', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Precision', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_precisions_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/precisions_realdata.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/precisions_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, recalls, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('Recall', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('Recall', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_recalls_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/recalls_realdata.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/recalls_' + str(num_test) + '.png', format='PNG')
# plt.close()

# plt.figure(figsize = (8,5), dpi = 300)
# plt.plot(thresholds, F1s, linewidth = 2)
# plt.xlabel('Threshold', fontsize = 18)
# plt.ylabel('F1', fontsize = 18)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)
# plt.title('F1', fontsize = 18)
# # plt.savefig(local_dir + 'plots/' + name + '/6_F1s_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/F1s_realdata.png', format='PNG')
# # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/F1s_' + str(num_test) + '.png', format='PNG')
# plt.close()

# # ##### -------------------- GAUSSIAN PEAK POSITION TEST -------------------- #####

# # print('DOING PEAK POSITION TESTS')

# # # print(target[4])
# # # print(test_predictions[4])

# # thresholds = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])

# # # threshold = 0.2

# # iterate = np.arange(0,len(realtest_predictions),1)
# # s = 0

# # fig = plt.subplots(nrows = 3, ncols = 4, figsize = (18,14), dpi = 400)
# # # fig = plt.subplots(nrows = 3, ncols = 4, figsize = (18,14))
# # plt.suptitle('Target vs. Prediction Samples Off per Threshold', fontsize = 20)

# # for idx in range(len(thresholds)):
    
# #     threshold = thresholds[idx]

# #     pred_binary = np.zeros(len(realtest_predictions))
# #     iterate = np.arange(0,len(realtest_predictions),1)
    
# #     for k in iterate:
# #         i = np.where(realtest_predictions[k] >= threshold)[0]
# #         if len(i) == 0:
# #             pred_binary[k] = 0
# #         elif len(i) > 0:
# #             pred_binary[k] = 1
    
# #     # print('Predictions: ')
# #     # print(pred_binary)
    
# #     # Convert the target arrays to single ones and zeroes
    
# #     targ_binary = np.zeros(len(gauss_target)) # Need to make this ones at indices in rows_w_eqs
    
# #     for idx in range(len(targ_binary)):
        
# #         if idx in rows_w_eqs:
            
# #             targ_binary[idx] = 1
    
# #     # print('Targets: ')
# #     # print(targ_binary)
    
# #     signals = []
    
# #     for i in iterate:
# #         pred = pred_binary[i]
# #         targ = targ_binary[i]
        
# #         # print(pred)
# #         # print(targ)
        
# #         if pred == 1 and targ == 1: # True positive, there was a signal and it found it
# #             signals.append(i) # Grab index from list of events that are correct and have a pick
# #         else:
# #             pass
    
# #     # print(signals)
    
# #     samples_off_list = []
    
# #     for index in signals:
        
# #         # Find the peak and then the index where that peak is and compare 
        
# #         # print('----------------------')
# #         # print('Signal number: ' + str(index))
        
# #         target_max_idx = np.argmax(gauss_target[index])
# #         # print('Target: ' + str(target_max_idx))
        
# #         pred_max_idx = np.argmax(realtest_predictions[index])
# #         # print('Prediction: ' + str(pred_max_idx))
        
# #         samples_off = np.abs(pred_max_idx - target_max_idx)
# #         # print('Samples off: ' + str(samples_off))
# #         samples_off_list.append(samples_off)
        
# #     # print(samples_off_list)
    
# #     plt.subplot(3,4,idx+1)
# #     plt.hist(samples_off_list, bins=128, range=(0,128), label = 'Threshold: ' + str(threshold))
# #     plt.xlim(0,128)
# #     plt.ylim(0,300)
# #     plt.legend()
# #     plt.grid(which = 'major', color = 'lightgray')
# #     plt.subplots_adjust(hspace = 0, wspace = 0)

# #     if idx == 0:
# #         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
    
# #     elif idx == 4:
# #         plt.ylabel('Number of examples in bin')
# #         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
# #         plt.yticks([0, 50, 100, 150, 200, 250])
        
# #     elif idx == 8:
# #         plt.yticks([0, 50, 100, 150, 200, 250])
        
# #     elif idx == 9:
# #         plt.xlabel('Numbers of samples off target position')
# #         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
        
# #     elif idx == 10:     
# #         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
        
# #     else:
# #         plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
# #         plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
    
# #     plt.subplot(3,4,12)
# #     plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
# #     plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)

# # # plt.savefig(local_dir + 'plots/' + name + '/7_histogram.png', format='PNG')
# # plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/10_realdata_peakpos_stats/histogram.png', format='PNG')
# # # plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/8_peakpos_stats/histogram.png', format='PNG')
# # plt.close()

# # ##### -------------------- METADATA ANALYSIS -------------------- #####

# # print('DOING REAL METADATA ANALYSIS')

# # print(stack_data.shape) # the data
# # print(norm_real_meta_data.shape) # the metadata
# # print(realtest_predictions.shape) # the model's predictions about the data

# # # np.save('batch_out_11_10_21.npy', batch_out)  
# # # np.save('test_preds_11_10_21.npy', test_predictions)  
# # # np.save('target_11_10_21.npy', target)

# # # print(batch_out[0])
# # # print(metadata[0][0])
# # # print(test_predictions[0])

# # # print(metadata)

# # zeros = np.zeros((realtest_predictions.shape[0],1))
# # analysis_array = np.c_[norm_real_meta_data,zeros]

# # # Metadata columns: station, date, start time, end time, counter, gauss position, pgd, SNR N component, SNR E, SNR Z

# # # print(analysis_array.shape)

# # for i in range(len(stack_data)):
    
# #     # print(i)
    
# #     # print(metadata[i])

# #     if norm_real_meta_data[i][5] == 'nan':
# #         # print(str(i) + ' is not an earthquake')
# #         # analysis_array[i][3] = 'nan'
        
# #         threshold = 0.16
        
# #         # True positive, true negative, false positive, false negative
        
# #         # print('Threshold: ' + str(threshold))
    
# #         # Convert the predictions arrays to single ones and zeroes
        
# #         p = np.where(test_predictions[i] >= threshold)[0]
# #         if len(p) == 0:
# #             pred_binary = 0
# #         elif len(p) > 0:
# #             pred_binary = 1
        
# #         # if i == 0:
# #         #     print('Prediction: ')
# #         #     print(pred_binary)
        
# #         # # Convert the target arrays to single ones and zeroes
        
# #         t = np.where(target[i] > 0)[0]
# #         if len(t) == 0:
# #             targ_binary = 0
# #         elif len(t) > 0:
# #             targ_binary = 1
        
# #         # if i == 0:
# #         #     print('Target: ')
# #         #     print(targ_binary)
        
# #         pred = pred_binary
# #         targ = targ_binary
        
# #         if pred == targ: # add one to list of correct predictions if matching
# #             # correct_preds.append(1)
            
# #             if pred == 1 and targ == 1:
# #                 result = 'true pos'
# #             elif pred == 0 and targ == 0:
# #                 result = 'true neg'
            
# #         elif pred != targ: # add ones to list of incorrect predictions if not matching
# #             # wrong_preds.append(1)
            
# #             if pred == 1 and targ == 0:
# #                 result = 'false pos'
# #             elif pred == 0 and targ == 1:
# #                 result = 'false neg'
        
# #         analysis_array[i][3] = result
    
# #     else:
# #         # print(str(i) + ' is an earthquake')
        
# #         rupt_num = metadata[i][0]
# #         station = metadata[i][1]
# #         mag = metadata[i][2]
        
# #         # print(rupt_num)
# #         # print(station)
# #         # print(mag)
        
# #         # print(batch_out[i])
# #         # print(test_predictions[i])
# #         # plt.plot(test_predictions[i])
# #         # plt.show()
        
# #         threshold = 0.2
        
# #         # True positive, true negative, false positive, false negative
        
# #         # print('Threshold: ' + str(threshold))
    
# #         # Convert the predictions arrays to single ones and zeroes
        
# #         p = np.where(test_predictions[i] >= threshold)[0]
# #         if len(p) == 0:
# #             pred_binary = 0
# #         elif len(p) > 0:
# #             pred_binary = 1
        
# #         # if i == 0:
# #         #     print('Prediction: ')
# #         #     print(pred_binary)
        
# #         # # Convert the target arrays to single ones and zeroes
        
# #         t = np.where(target[i] > 0)[0]
# #         if len(t) == 0:
# #             targ_binary = 0
# #         elif len(t) > 0:
# #             targ_binary = 1
        
# #         # if i == 0:
# #         #     print('Target: ')
# #         #     print(targ_binary)
        
# #         pred = pred_binary
# #         targ = targ_binary
        
# #         if pred == targ: # add one to list of correct predictions if matching
# #             # correct_preds.append(1)
            
# #             if pred == 1 and targ == 1:
# #                 result = 'true pos'
# #             elif pred == 0 and targ == 0:
# #                 result = 'true neg'
            
# #         elif pred != targ: # add ones to list of incorrect predictions if not matching
# #             # wrong_preds.append(1)
            
# #             if pred == 1 and targ == 0:
# #                 result = 'false pos'
# #             elif pred == 0 and targ == 1:
# #                 result = 'false neg'
        
# #         analysis_array[i][3] = result
    
# # print(analysis_array)
# # print(analysis_array.shape)
            
# # # # np.save('/home/sdybing/GNSS_project/' + name + 'testing_for_analysis.npy', analysis_array) # VAL
# # np.save('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/testing_for_analysis.npy', analysis_array) # LAP        







            
            
            

            
            
            
            
            
            





            
            
            

            
            
            
            
            
            