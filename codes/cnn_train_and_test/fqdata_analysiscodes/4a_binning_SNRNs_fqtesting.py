#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:19:14 2022

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt
    
SNRN_list = []

SNRNs = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/shuffle_trainfull/SNRs_N_fakequakes_testing.npy')
print(SNRNs[0])
print(SNRNs.shape)

i = np.where(SNRNs == '0.0')[0]
# print(i)

np.savetxt('zeros_SNRNs_idxs.txt', i)

results = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/shuffle_trainfull/analysis_w_hypdist_result_waveformpgd.csv', dtype = str)
print(results[0])
print(len(results))

snr_215 = []
snr_151 = []
snr_105 = []
snr_050 = []
snr_005 = []
snr_051 = []
snr_115 = []
snr_152 = []
snr_225 = []
snr_253 = []
snr_335 = []

true_pos = []
false_neg = []

dots = []

skips = []

for i in range(len(SNRNs)):
    
    logSNRN = np.log10(float(SNRNs[i]))
    result = results[i][4]
    
    if SNRNs[i] == 'nan' or SNRNs[i] == '0.0':
        
        # pass
        skips.append('skip')
        # SNRN_list.append('nan')
    
    else:
        
        # print(logSNRN)
        
        if logSNRN <= -1.5:
            snr_215.append(i)
        
        elif logSNRN > -1.5 and logSNRN <= -1:
            snr_151.append(i)
        
        elif logSNRN > -1 and logSNRN <= -0.5:
            snr_105.append(i)
        
        elif logSNRN > -0.5 and logSNRN <= 0:
            snr_050.append(i)
            
        elif logSNRN > 0 and logSNRN <= 0.5:
            snr_005.append(i)
            
        elif logSNRN > 0.5 and logSNRN <= 1:
            snr_051.append(i)
            
        elif logSNRN > 1 and logSNRN <= 1.5:
            snr_115.append(i)
            
        elif logSNRN > 1.5 and logSNRN <= 2:
            snr_152.append(i)
            
        elif logSNRN > 2 and logSNRN <= 2.5:
            snr_225.append(i)
            
        elif logSNRN > 2.5 and logSNRN <= 3:
            snr_253.append(i)
            
        elif logSNRN > 3 and logSNRN <= 3.5:
            snr_335.append(i)
        
        if result == 'true_pos': # predicted 1, target 1
            dot = 1
            dots.append(dot)
            true_pos.append(1)
        
        elif result == 'false_neg': # predicted 0, target 1
            dot = 0
            dots.append(dot)
            false_neg.append(1)
        
        else:
            pass
       
        # SNRN_list.append(float(SNRN))
    
# print(len(SNRN_list))
# logminSNRN = np.log10(min(SNRN_list))
# logmaxSNRN = np.log10(max(SNRN_list))

# print(logminSNRN)
# print(logmaxSNRN)

# plt.hist(SNRN_list, bins = 50)

# Now need to bring in the true pos etc. info

dots = np.asarray(dots)
print('Skips: ' + str(len(skips)))

number_correct = len(true_pos)
number_incorrect = len(false_neg)
total_number = number_correct + number_incorrect

print('True positive: ' + str(number_correct))
print('False negative: ' + str(number_incorrect))
print('Total number: ' + str(total_number))
# print('Accuracy for real earthquakes: ' + str(100*(number_correct/total_number)) + '%')

# snr_215 = []
# snr_151 = []
# snr_105 = []
# snr_050 = []
# snr_005 = []
# snr_051 = []
# snr_115 = []
# snr_152 = []
# snr_225 = []
# snr_253 = []
# snr_335 = []
# snr_354 = []

print(len(snr_215) + len(snr_151) + len(snr_105) + len(snr_050) + len(snr_005) + len(snr_051) + len(snr_115) + len(snr_152) + len(snr_225) + len(snr_253) + len(snr_335))

# Calculating accuracies for bins

accuracy_percentages = []

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN below -1.5')
print('-------------------------------------------------------------')

truepos_215 = []
falseneg_215 = []

for index in snr_215:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_215.append(1)
    elif result == 'false_neg':
        falseneg_215.append(1)
        
number_correct_215 = len(truepos_215)
number_incorrect_215 = len(falseneg_215)
total_number_215 = number_correct_215 + number_incorrect_215

print('True positive: ' + str(number_correct_215))
print('False negative: ' + str(number_incorrect_215))
print('Total number: ' + str(total_number_215))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_215/total_number_215)) + '%')

accuracy_215 = 100*(number_correct_215/total_number_215)
accuracy_percentages.append(accuracy_215)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN -1.5 to -1')
print('-------------------------------------------------------------')

truepos_151 = []
falseneg_151 = []

for index in snr_151:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_151.append(1)
    elif result == 'false_neg':
        falseneg_151.append(1)
        
number_correct_151 = len(truepos_151)
number_incorrect_151 = len(falseneg_151)
total_number_151 = number_correct_151 + number_incorrect_151

print('True positive: ' + str(number_correct_151))
print('False negative: ' + str(number_incorrect_151))
print('Total number: ' + str(total_number_151))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_151/total_number_151)) + '%')

accuracy_151 = 100*(number_correct_151/total_number_151)
accuracy_percentages.append(accuracy_151)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN -1 to -0.5')
print('-------------------------------------------------------------')

truepos_105 = []
falseneg_105 = []

for index in snr_105:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_105.append(1)
    elif result == 'false_neg':
        falseneg_105.append(1)
        
number_correct_105 = len(truepos_105)
number_incorrect_105 = len(falseneg_105)
total_number_105 = number_correct_105 + number_incorrect_105

print('True positive: ' + str(number_correct_105))
print('False negative: ' + str(number_incorrect_105))
print('Total number: ' + str(total_number_105))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_105/total_number_105)) + '%')

accuracy_105 = 100*(number_correct_105/total_number_105)
accuracy_percentages.append(accuracy_105)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN -0.5 to 0')
print('-------------------------------------------------------------')

truepos_050 = []
falseneg_050 = []

for index in snr_050:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_050.append(1)
    elif result == 'false_neg':
        falseneg_050.append(1)
        
number_correct_050 = len(truepos_050)
number_incorrect_050 = len(falseneg_050)
total_number_050 = number_correct_050 + number_incorrect_050

print('True positive: ' + str(number_correct_050))
print('False negative: ' + str(number_incorrect_050))
print('Total number: ' + str(total_number_050))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_050/total_number_050)) + '%')

accuracy_050 = 100*(number_correct_050/total_number_050)
accuracy_percentages.append(accuracy_050)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 0 to 0.5')
print('-------------------------------------------------------------')

truepos_005 = []
falseneg_005 = []

for index in snr_005:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_005.append(1)
    elif result == 'false_neg':
        falseneg_005.append(1)
        
number_correct_005 = len(truepos_005)
number_incorrect_005 = len(falseneg_005)
total_number_005 = number_correct_005 + number_incorrect_005

print('True positive: ' + str(number_correct_005))
print('False negative: ' + str(number_incorrect_005))
print('Total number: ' + str(total_number_005))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_005/total_number_005)) + '%')

accuracy_005 = 100*(number_correct_005/total_number_005)
accuracy_percentages.append(accuracy_005)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 0.5 to 1')
print('-------------------------------------------------------------')

truepos_051 = []
falseneg_051 = []

for index in snr_051:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_051.append(1)
    elif result == 'false_neg':
        falseneg_051.append(1)
        
number_correct_051 = len(truepos_051)
number_incorrect_051 = len(falseneg_051)
total_number_051 = number_correct_051 + number_incorrect_051

print('True positive: ' + str(number_correct_051))
print('False negative: ' + str(number_incorrect_051))
print('Total number: ' + str(total_number_051))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_051/total_number_051)) + '%')

accuracy_051 = 100*(number_correct_051/total_number_051)
accuracy_percentages.append(accuracy_051)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 1 to 1.5')
print('-------------------------------------------------------------')

truepos_115 = []
falseneg_115 = []

for index in snr_115:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_115.append(1)
    elif result == 'false_neg':
        falseneg_115.append(1)
        
number_correct_115 = len(truepos_115)
number_incorrect_115 = len(falseneg_115)
total_number_115 = number_correct_115 + number_incorrect_115

print('True positive: ' + str(number_correct_115))
print('False negative: ' + str(number_incorrect_115))
print('Total number: ' + str(total_number_115))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_115/total_number_115)) + '%')

accuracy_115 = 100*(number_correct_115/total_number_115)
accuracy_percentages.append(accuracy_115)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 1.5 to 2')
print('-------------------------------------------------------------')

truepos_152 = []
falseneg_152 = []

for index in snr_152:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_152.append(1)
    elif result == 'false_neg':
        falseneg_152.append(1)
        
number_correct_152 = len(truepos_152)
number_incorrect_152 = len(falseneg_152)
total_number_152 = number_correct_152 + number_incorrect_152

print('True positive: ' + str(number_correct_152))
print('False negative: ' + str(number_incorrect_152))
print('Total number: ' + str(total_number_152))
print('Accuracy for real earthquakes: ' + str(1520*(number_correct_152/total_number_152)) + '%')

accuracy_152 = 1520*(number_correct_152/total_number_152)
accuracy_percentages.append(accuracy_152)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 2 to 2.5')
print('-------------------------------------------------------------')

truepos_225 = []
falseneg_225 = []

for index in snr_225:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_225.append(1)
    elif result == 'false_neg':
        falseneg_225.append(1)
        
number_correct_225 = len(truepos_225)
number_incorrect_225 = len(falseneg_225)
total_number_225 = number_correct_225 + number_incorrect_225

print('True positive: ' + str(number_correct_225))
print('False negative: ' + str(number_incorrect_225))
print('Total number: ' + str(total_number_225))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_225/total_number_225)) + '%')

accuracy_225 = 100*(number_correct_225/total_number_225)
accuracy_percentages.append(accuracy_225)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 2.5 to 3')
print('-------------------------------------------------------------')

truepos_253 = []
falseneg_253 = []

for index in snr_253:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_253.append(1)
    elif result == 'false_neg':
        falseneg_253.append(1)
        
number_correct_253 = len(truepos_253)
number_incorrect_253 = len(falseneg_253)
total_number_253 = number_correct_253 + number_incorrect_253

print('True positive: ' + str(number_correct_253))
print('False negative: ' + str(number_incorrect_253))
print('Total number: ' + str(total_number_253))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_253/total_number_253)) + '%')

accuracy_253 = 100*(number_correct_253/total_number_253)
accuracy_percentages.append(accuracy_253)

print(' ')
print('-------------------------------------------------------------')
print('Log SNRN 3 to 3.5')
print('-------------------------------------------------------------')

truepos_335 = []
falseneg_335 = []

for index in snr_335:
    
    # print(index)
    event_row = results[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_335.append(1)
    elif result == 'false_neg':
        falseneg_335.append(1)
        
number_correct_335 = len(truepos_335)
number_incorrect_335 = len(falseneg_335)
total_number_335 = number_correct_335 + number_incorrect_335

print('True positive: ' + str(number_correct_335))
print('False negative: ' + str(number_incorrect_335))
print('Total number: ' + str(total_number_335))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_335/total_number_335)) + '%')

accuracy_335 = 100*(number_correct_335/total_number_335)
accuracy_percentages.append(accuracy_335)

# Plotting accuracies by bin

bar_positions = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

print(len(bar_positions))
print(len(accuracy_percentages))

plt.figure(dpi=300)
# plt.figure()
plt.bar(bar_positions, accuracy_percentages, width = 0.5, align = 'edge', edgecolor = 'black')
# plt.axvline(x = np.log10(0.02), color = 'darkorange', linewidth = 3)
plt.xlim(-2,3.5)
plt.ylim(0,100)
plt.ylabel('Accuracy (%)')
plt.xlabel('Log SNR (North-South Component')
plt.title('Algorithm accuracy (threshold = 0.615)')
plt.suptitle('Log SNRN')

# plt.show()
# plt.savefig('binned_logSNRN_acc.png', format='PNG')
plt.close()

np.savetxt('N_barpos.txt', bar_positions)
np.savetxt('N_accper.txt', accuracy_percentages)





