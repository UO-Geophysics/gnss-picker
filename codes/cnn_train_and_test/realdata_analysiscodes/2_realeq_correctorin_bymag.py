#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:09:23 2022

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

correct_eq_inds = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/newtrain_march/more_realdata_norm_testing/correct_indices.txt', dtype = 'int') # big dataset
print(correct_eq_inds.shape) # In big real data dataset, there should be 2557 earthquakes. Need histogram of magnitudes of those
# How many correct is this 156 - average SNR of the ones correct? Histogram?
# How spot on are the pick locations?

rows_w_eqs = np.load('rows_w_eqs_rembad.npy')
print(rows_w_eqs.shape)

# Checked to make sure none of the removed bad ones were in correct eq inds - they aren't

# metadata = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pgd_snr.npy')

############## for removing bad rows

# mags = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rwe_mags.npy')
# print(mags.shape)

# # bad_N = np.load('toomanyzeros_N.npy')
# # bad_E = np.load('toomanyzeros_E.npy')
# # bad_Z = np.load('toomanyzeros_Z.npy')

# # comb = np.concatenate((bad_N, bad_E, bad_Z))

# # remove_dups = list(set(comb))
# # remove_dups = np.sort(remove_dups)

# # deleted_eqs = []

# # rows_w_eqs = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rowsweqs.npy')
# # print(rows_w_eqs.shape)

# # for idx in range(len(rows_w_eqs)):
# #     #print(rows_w_eqs[idx])
    
# #     for i in range(len(remove_dups)):
# #         #print(remove_dups[i])
        
# #         if remove_dups[i] == rows_w_eqs[idx]:
# #             #print(rows_w_eqs[idx])
# #             #print(remove_dups[i])
# #             deleted_eqs.append(idx)

# # print(deleted_eqs)
# deleted_eqs = np.loadtxt('rwe_deleted_eqs_list.txt')
# print(deleted_eqs)
# deleted_eqs = np.int_(deleted_eqs)
# print(deleted_eqs)

# mags_rembad = np.delete(mags, deleted_eqs, axis = 0)
# # # mags = mags_rembad
# print(mags_rembad.shape)
# np.save('realdata_mags_rembad.npy', mags_rembad)

############## 

mags = np.load('realdata_mags_rembad.npy')

metadata = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/real_metadata_w_gauss_pgd_snr.npy')
print(metadata.shape)
metadata_eqs = metadata[rows_w_eqs]
print(metadata_eqs.shape)
pgds_m = metadata_eqs[:,6]
print(pgds_m)

snrsn = metadata_eqs[:,7]
snrse = metadata_eqs[:,8]
snrsz = metadata_eqs[:,9]

# Columns: station, date, start time, end time, counter, gauss position, pgd, SNR N component, SNR E, SNR Z

# mags_fake = np.random.rand(2352)
# print(mags_fake.shape)

# mags_fake2 = np.random.rand(156)
# print(mags_fake2.shape)

# plt.scatter(rows_w_eqs, mags_fake)
# plt.scatter(correct_eq_inds, mags_fake2)

matching_rows_idxinrwe = []
count = 0

for j in range(len(correct_eq_inds)):
    
    correct_row = correct_eq_inds[j]
    # print(correct_row)
    
    for i in range(len(rows_w_eqs)):
        
        row = rows_w_eqs[i]
        #print(row)
        
        if correct_row == row:
            
            # print(correct_row)
            # print(row)
            matching_rows_idxinrwe.append(i)
            count += 1

print(count)
print(matching_rows_idxinrwe)

incorrect_eq_inds = np.delete(rows_w_eqs, matching_rows_idxinrwe, axis = 0)

mags_correct = mags[matching_rows_idxinrwe].astype(float)
mags_incorrect = (np.delete(mags, matching_rows_idxinrwe, axis = 0)).astype(float)

pgds_m_correct = pgds_m[matching_rows_idxinrwe].astype(float)
pgds_m_incorrect = (np.delete(pgds_m, matching_rows_idxinrwe, axis = 0)).astype(float)

snrsn_correct = snrsn[matching_rows_idxinrwe].astype(float)
snrsn_incorrect = (np.delete(snrsn, matching_rows_idxinrwe, axis = 0)).astype(float)

snrse_correct = snrse[matching_rows_idxinrwe].astype(float)
snrse_incorrect = (np.delete(snrse, matching_rows_idxinrwe, axis = 0)).astype(float)

snrsz_correct = snrsz[matching_rows_idxinrwe].astype(float)
snrsz_incorrect = (np.delete(snrsz, matching_rows_idxinrwe, axis = 0)).astype(float)

print(len(mags_correct))
print(len(mags_incorrect))
print(len(incorrect_eq_inds))
print(len(pgds_m_correct))
print(len(pgds_m_incorrect))

plt.figure(figsize = (10,5), dpi=300)
# plt.figure(figsize = (10,7))
plt.scatter(incorrect_eq_inds, mags_incorrect, label = 'Missed (2,196)', color = '#2DADB4', alpha = 0.5)
plt.scatter(correct_eq_inds, mags_correct, label = 'Picked (156)', color = '#001528', alpha = 0.5)
plt.ylabel('Magnitude', fontsize = 16)
plt.xlabel('Sample index (those with earthquakes only)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
#plt.title('Picked vs. missed earthquake magnitudes', fontsize = 15)
plt.title('Picked vs. Missed Earthquakes by Magnitude', fontsize = 17)
plt.legend(loc = (0.67, 0.52), fontsize = 16)
# plt.show()
plt.savefig('pickmiss_mag.png', type = 'PNG')
plt.close()

plt.figure(figsize = (10,5), dpi=300)
# plt.figure(figsize = (10,7))
plt.scatter(incorrect_eq_inds, pgds_m_incorrect*100, label = 'Missed (2,196)', color = '#2DADB4', alpha = 0.5)
plt.scatter(correct_eq_inds, pgds_m_correct*100, label = 'Picked (156)', color = '#001528', alpha = 0.5)
plt.ylabel('PGD (cm)', fontsize = 16)
plt.xlabel('Sample index (those with earthquakes only)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.yscale('log')
#plt.title('Picked vs. missed earthquake magnitudes', fontsize = 15)
plt.title('Picked vs. Missed Earthquakes by Peak Ground Displacement (cm)', fontsize = 17)
plt.legend(loc = 'upper left', fontsize = 16)
# plt.show()
plt.savefig('pickmiss_pgd.png', type = 'PNG')
plt.close()


plt.figure(figsize = (10,5), dpi=300)
# plt.figure(figsize = (10,7))
plt.scatter(incorrect_eq_inds, snrsn_incorrect, label = 'Missed (2,196)', color = '#2DADB4', alpha = 0.5)
plt.scatter(correct_eq_inds, snrsn_correct, label = 'Picked (156)', color = '#001528', alpha = 0.5)
plt.ylabel('SNR', fontsize = 16)
plt.xlabel('Sample index (those with earthquakes only)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.yscale('log')
#plt.title('Picked vs. missed earthquake magnitudes', fontsize = 15)
plt.title('Picked vs. Missed Earthquakes by Signal to Noise Ratio - N component', fontsize = 17)
plt.legend(loc = 'lower left', fontsize = 16)
# plt.show()
plt.savefig('pickmiss_SNRN.png', type = 'PNG')
plt.close()

    
plt.figure(figsize = (10,5), dpi=300)
# plt.figure(figsize = (10,7))
plt.scatter(incorrect_eq_inds, snrse_incorrect, label = 'Missed (2,196)', color = '#2DADB4', alpha = 0.5)
plt.scatter(correct_eq_inds, snrse_correct, label = 'Picked (156)', color = '#001528', alpha = 0.5)
plt.ylabel('SNR', fontsize = 16)
plt.xlabel('Sample index (those with earthquakes only)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.yscale('log')
#plt.title('Picked vs. missed earthquake magnitudes', fontsize = 15)
plt.title('Picked vs. Missed Earthquakes by Signal to Noise Ratio - E component', fontsize = 17)
plt.legend(loc = 'lower left', fontsize = 16)
# plt.show()    
plt.savefig('pickmiss_SNRE.png', type = 'PNG')
plt.close()
    
plt.figure(figsize = (10,5), dpi=300)
# plt.figure(figsize = (10,7))
plt.scatter(incorrect_eq_inds, snrsz_incorrect, label = 'Missed (2,196)', color = '#2DADB4', alpha = 0.5)
plt.scatter(correct_eq_inds, snrsz_correct, label = 'Picked (156)', color = '#001528', alpha = 0.5)
plt.ylabel('SNR', fontsize = 16)
plt.xlabel('Sample index (those with earthquakes only)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.yscale('log')
#plt.title('Picked vs. missed earthquake magnitudes', fontsize = 15)
plt.title('Picked vs. Missed Earthquakes by Signal to Noise Ratio - Z component', fontsize = 17)
plt.legend(loc = 'lower left', fontsize = 16)
# plt.show() 
plt.savefig('pickmiss_SNRZ.png', type = 'PNG')
plt.close()   
    
    
    
