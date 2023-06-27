#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 01:02:16 2022

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

### REMOVING BAD DATA ###

realtest_predictions = np.load('realtest_predictions.npy')
gauss_target = np.load('gauss_target.npy')
stack_data = np.load('stack_data.npy')
rows_w_eqs = np.load('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/More_RealData/rowsweqs.npy')

name = 'shuffle_trainfull'

bad_N = np.load('toomanyzeros_N.npy')
bad_E = np.load('toomanyzeros_E.npy')
bad_Z = np.load('toomanyzeros_Z.npy')

# print(len(bad_N))
# print(len(bad_E))
# print(len(bad_Z))

#print(len(bad_N) + len(bad_E) + len(bad_Z))

comb = np.concatenate((bad_N, bad_E, bad_Z))
#print(len(comb))

remove_dups = list(set(comb))
remove_dups = np.sort(remove_dups)
np.save('remove_dups_realdata.npy', remove_dups)
#print(len(remove_dups))

# stack_data_rembad = np.delete(stack_data, remove_dups, axis = 0)
# # print(len(stack_data))
# # print(len(stack_data_rembad))

# realtest_predictions_rembad = np.delete(realtest_predictions, remove_dups, axis = 0)
# gauss_target_rembad = np.delete(gauss_target, remove_dups, axis = 0)

# deleted_eqs = []

# for idx in range(len(rows_w_eqs)):
#     #print(rows_w_eqs[idx])
    
#     for i in range(len(remove_dups)):
#         #print(remove_dups[i])
        
#         if remove_dups[i] == rows_w_eqs[idx]:
#             #print(rows_w_eqs[idx])
#             #print(remove_dups[i])
#             deleted_eqs.append(idx)

# #print(deleted_eqs)

# #print(remove_dups)
# #print(rows_w_eqs[2527])

# rows_w_eqs_rembad = np.delete(rows_w_eqs, deleted_eqs, axis = 0)

# print(len(stack_data))
# print(len(stack_data_rembad))

# print(len(realtest_predictions))
# print(len(realtest_predictions_rembad))

# print(len(gauss_target))
# print(len(gauss_target_rembad))

# print(len(rows_w_eqs))
# print(len(rows_w_eqs_rembad))

# np.save('stack_data_rembad.npy', stack_data_rembad)
# np.save('realtest_predictions_rembad.npy', realtest_predictions_rembad)
# np.save('gauss_target_rembad.npy', gauss_target_rembad)
# np.save('rows_w_eqs_rembad.npy', rows_w_eqs_rembad)

# Finding the too many zeros

# # i = np.random.randint(0, high = len(stack_data))
# # plt.plot(stack_data[i,:,0])

# sample = stack_data[57098:57102,:,0]
# samp_diff = np.diff(sample)
# samp_diff2 = np.diff(samp_diff)

# #print(samp_diff.shape)

# #a = 57098
# #a = 0

# diff_N = np.diff(stack_data[:,:,0])
# diff_E = np.diff(stack_data[:,:,1])
# diff_Z = np.diff(stack_data[:,:,2])

# diff2_N = np.diff(diff_N)
# diff2_E = np.diff(diff_E)
# diff2_Z = np.diff(diff_Z)

# toomanyzeros_N = []
# toomanyzeros_E = []
# toomanyzeros_Z = []

# for k in range(len(stack_data)):
#     #a += 1
#     #print(a)
#     b = 0
#     for j in range(len(diff2_N[k])):
#         point = diff2_N[k][j]
#         check = np.isclose(point, 0, rtol = 1e-05)
#         if check == True:
#             b += 1
#     #print(b)
#     if b >= 60:
#         #print('Too many zeros!')
#         toomanyzeros_N.append(k)

# print(toomanyzeros_N)

# for k in range(len(stack_data)):
#     #a += 1
#     #print(a)
#     b = 0
#     for j in range(len(diff2_E[k])):
#         point = diff2_E[k][j]
#         check = np.isclose(point, 0, rtol = 1e-05)
#         if check == True:
#             b += 1
#     print(b)
#     if b >= 60:
#         print('Too many zeros!')
#         toomanyzeros_E.append(k)

# print(toomanyzeros_E)

# for k in range(len(stack_data)):
#     #a += 1
#     #print(a)
#     b = 0
#     for j in range(len(diff2_Z[k])):
#         point = diff2_Z[k][j]
#         check = np.isclose(point, 0, rtol = 1e-05)
#         if check == True:
#             b += 1
#     print(b)
#     if b >= 60:
#         print('Too many zeros!')
#         toomanyzeros_Z.append(k)

# print(toomanyzeros_Z)

# np.save('toomanyzeros_N.npy', np.array(toomanyzeros_N))
# np.save('toomanyzeros_E.npy', np.array(toomanyzeros_E))
# np.save('toomanyzeros_Z.npy', np.array(toomanyzeros_Z))

# for k in range(len(samp_diff)):
#     a += 1
#     print(a)
#     #print(samp_diff2[k])
#     b = 0
#     for j in range(len(samp_diff2[k])):
#         point = samp_diff2[k][j]
#         check = np.isclose(point, 0, rtol = 1e-05)
#         if check == True:
#             b += 1
#     print(b)
#     if b >= 60:
#         print('Too many zeros!')
#         toomanyzeros.append(a)
#     #print(samp_diff[k])
#     fig, ax1 = plt.subplots()
#     ax1.plot(sample[k], color = 'C0', label = 'Data')
#     #plt.show()
#     ax2 = ax1.twinx()
#     ax2.plot(samp_diff[k], color = 'C1', label = 'First derivative')
#     ax2.plot(samp_diff2[k], color = 'C2', label = 'Second derivative')
#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2)
#     plt.show()
#     # plt.savefig('derivtestplot' + str(a) + '.png')




### LOAD DATA THAT HAS BAD ONES REMOVED ###

realtest_predictions = np.load('realtest_predictions_rembad.npy')
gauss_target = np.load('gauss_target_rembad.npy')
stack_data = np.load('stack_data_rembad.npy')
rows_w_eqs = np.load('rows_w_eqs_rembad.npy')

name = 'shuffle_trainfull'

# ##### -------------------- CLASSIFICATION TESTS -------------------- #####

print('DOING CLASSIFICATION TESTS ON REAL DATA')

# Decision threshold evaluation

thresholds = np.arange(0, 1.005, 0.005)
# thresholds = np.arange(0, 1.1, 0.1)
# thresholds = np.arange(0, 1, 0.1)
test_thresholds = [0.6]

# Use np.where to see whether anywhere in test_predictions is > threshold
# If there is a value that's >, the 'result' of the array is 1. If not 0
# Then compare these 1s and 0s to the target array value for PAR

accuracies = []
accuracies_per = []
precisions = []
recalls = []
F1s = []

TP_pert = []
TN_pert = []
FP_pert = []
FN_pert = []

for threshold in thresholds:
    
    print('-------------------------------------------------------------')
    print('Threshold: ' + str(threshold))
    print('-------------------------------------------------------------')
    print(' ')
    
    # Convert the predictions arrays to single ones and zeroes
    
    pred_binary = np.zeros(len(realtest_predictions))
    iterate = np.arange(0,len(realtest_predictions),1)
    
    for k in iterate:
        # print('Prediction: ' + str(realtest_predictions[k]))
        i = np.where(realtest_predictions[k] >= threshold)[0]
        # print(i)
        if len(i) == 0:
            pred_binary[k] = 0
        elif len(i) > 0:
            pred_binary[k] = 1
    
    print('Predictions: ')
    print(pred_binary)
    print(pred_binary.shape)
    
    # Convert the target arrays to single ones and zeroes
    
    targ_binary = np.zeros(len(gauss_target)) # Need to make this ones at indices in rows_w_eqs
    
    for idx in range(len(targ_binary)):
        
        if idx in rows_w_eqs:
            
            targ_binary[idx] = 1
    
    print('Targets: ')
    print(targ_binary)
    
    # Calculating the accuracy, precision, recall, and F1
    
    num_preds = len(realtest_predictions) # total number of predictions. Amanda did 50
    correct_preds = []
    wrong_preds = []
    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []
    
    for i in iterate:
        
        pred = pred_binary[i]
        targ = targ_binary[i]
        
        if pred == targ: # add one to list of correct predictions if matching
            correct_preds.append(1)
            
            if pred == 1 and targ == 1:
                true_pos.append(1)
            elif pred == 0 and targ == 0:
                true_neg.append(1)
            
        elif pred != targ: # add ones to list of incorrect predictions if not matching
            wrong_preds.append(1)
            
            if pred == 1 and targ == 0:
                false_pos.append(1)
            elif pred == 0 and targ == 1:
                false_neg.append(1)
    
    num_correct_preds = len(correct_preds)
    num_wrong_preds = len(wrong_preds)
    num_true_pos = len(true_pos)
    num_true_neg = len(true_neg)
    num_false_pos = len(false_pos)
    num_false_neg = len(false_neg)
    
    TP_pert.append(num_true_pos)
    TN_pert.append(num_true_neg)
    FP_pert.append(num_false_pos)
    FN_pert.append(num_false_neg)
    
    print('Threshold: ' + str(threshold))
    print('Correct preds: ' + str(num_correct_preds))
    print('Wrong preds: ' + str(num_wrong_preds))
    print('True pos: ' + str(num_true_pos))
    print('True neg: ' + str(num_true_neg))
    print('False pos: ' + str(num_false_pos))
    print('False neg: ' + str(num_false_neg))
    
    accuracy = num_correct_preds / num_preds
    accuracy_per = (num_correct_preds / num_preds) * 100
    print('Accuracy: ' + str(accuracy_per) + '%')
    
    if num_true_pos == 0  and num_false_pos == 0:
        precision = 0
    else:
        precision = num_true_pos / (num_true_pos + num_false_pos)
    
    if num_true_pos == 0 and num_false_neg == 0:
        recall = 0
    else:
        recall = num_true_pos / (num_true_pos + num_false_neg)
    
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * ((precision * recall) / (precision + recall))
        
    
    
    accuracies.append(accuracy)
    accuracies_per.append(accuracy_per)
    precisions.append(precision)
    recalls.append(recall)
    F1s.append(F1)

print('Accuracies')
print(accuracies)
print('Precisions')
print(precisions)
print('Recalls')
print(recalls)
print('F1s')
print(F1s)

plt.figure(figsize = (8,6.5), dpi = 300)
ax1 = plt.subplot(111)

ax1.plot(thresholds, np.array(TN_pert), label = 'True neg', color = 'blue')
ax1.plot(thresholds, np.array(FP_pert), label = 'False pos', color = 'red')
ax1.set_xlim(0,0.1)
ax2 = ax1.twinx()
ax2.plot(thresholds, np.array(TP_pert), label = 'True pos', color = 'green', linestyle = 'dashed')
ax2.plot(thresholds, np.array(FN_pert), label = 'False neg', color = 'orange', linestyle = 'dashed')
ax2.set_ylim(0,3000)
ax1.set_ylim(0,1070000)
ax1.set_xlabel('Threshold', fontsize = 14)
ax1.set_ylabel('Number of cases - TN & FP', fontsize = 14)
ax2.set_ylabel('Number of cases - TP & FN', fontsize = 14)
ax1.tick_params(labelsize = 14)
ax2.tick_params(labelsize = 14)
ax1.legend(fontsize = 14)
ax2.legend(fontsize = 14)
plt.title('Result Case Count', fontsize = 16)
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/outcomecurves_rembad_realdata_cropped.png', format='PNG')
plt.close()

np.savetxt('rembad_realdata_TNnum_txt.txt', TN_pert)
np.savetxt('rembad_realdata_FPnum_txt.txt', FP_pert)
np.savetxt('rembad_realdata_TPnum_txt.txt', TP_pert)
np.savetxt('rembad_realdata_FNnum_txt.txt', FN_pert)

np.savetxt('rembad_realdata_accuracies_percentage_txt.txt', accuracies_per)
np.savetxt('rembad_realdata_thresholds_txt.txt', thresholds)

plt.figure(figsize = (8,5), dpi = 300)
#plt.figure(figsize = (8,5))
# plt.scatter(thresholds,accuracies)
plt.plot(thresholds, accuracies_per, linewidth = 2)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('Accuracy (%)', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,100)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Accuracy Percentage', fontsize = 18)
# plt.show()
# plt.savefig(local_dir + 'plots/' + name + '/6_accuracies_' + str(num_test) + '.png', format='PNG')
# plt.savefig('/Users/sydneydybing/Documents/AGU_2021/Figures/6_accuracies_' + str(num_test) + '.png', format='PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/accuracies_rembad_realdata.png', format='PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/accuracies_' + str(num_test) + '.png', format='PNG')
plt.close()

plt.figure(figsize = (8,5), dpi = 300)
#plt.figure(figsize = (8,5))
plt.plot(thresholds, precisions, linewidth = 2)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('Precision', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Precision', fontsize = 18)
# plt.show()
# plt.savefig(local_dir + 'plots/' + name + '/6_precisions_' + str(num_test) + '.png', format='PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/precisions_rembad_realdata.png', format='PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/precisions_' + str(num_test) + '.png', format='PNG')
plt.close()

plt.figure(figsize = (8,5), dpi = 300)
#plt.figure(figsize = (8,5))
plt.plot(thresholds, recalls, linewidth = 2)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('Recall', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Recall', fontsize = 18)
# plt.show()
# plt.savefig(local_dir + 'plots/' + name + '/6_recalls_' + str(num_test) + '.png', format='PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/recalls_rembad_realdata.png', format='PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/recalls_' + str(num_test) + '.png', format='PNG')
plt.close()

plt.figure(figsize = (8,5), dpi = 300)
#plt.figure(figsize = (8,5))
plt.plot(thresholds, F1s, linewidth = 2)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('F1', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('F1', fontsize = 18)
# plt.show()
# plt.savefig(local_dir + 'plots/' + name + '/6_F1s_' + str(num_test) + '.png', format='PNG')
plt.savefig('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/' + name + '/figures/9_realdata_classify_stats/F1s_rembad_realdata.png', format='PNG')
# plt.savefig('/home/sdybing/GNSS_project/' + name + '/figures/7_classify_stats/F1s_' + str(num_test) + '.png', format='PNG')
plt.close()




