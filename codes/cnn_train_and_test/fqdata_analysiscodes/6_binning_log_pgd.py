#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:37:35 2021

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/shuffle_trainfull/analysis_w_hypdist_result_waveformpgd.csv', dtype = str)

print(len(data))
print(data[-1])
# minmag = min(data[:,2])
# maxmag = max(data[:,2])
# minpgd = min(data[:,3])
# maxpgd = max(data[:,3])
# minhyp = min(data[:,10])
# maxhyp = max(data[:,10])

pgds = []

for idx in range(len(data)):
    
    pgd = data[idx,3]
    
    if pgd != 'nan':
        
        pgds.append(float(pgd))

print(len(pgds))
print(np.log10(float(min(pgds))))
print(np.log10(float(max(pgds))))

plt.figure(dpi=500)
plt.hist(pgds, bins=50)
# plt.xlim(-6,0)
plt.ylabel('Number of records in bin')
plt.xlabel('Peak ground displacement (m)')
plt.title('Distribution of PGDs in data')
# plt.show()
plt.savefig('waveformpgd_histogram.png', format='PNG')
plt.close()

plt.figure(dpi=500)
plt.hist(np.log10(pgds), bins=50)
plt.xlim(-6,1)
plt.ylim(0,100)
plt.ylabel('Number of records in bin')
plt.xlabel('Log peak ground displacement (m)')
plt.title('Distribution of log PGDs in data')
# plt.show()
plt.savefig('waveform_logpgd_histogram_to100.png', format='PNG')
plt.close()

# # This CSV has columns for rupture number, station, magnitude, pgd, classification 
# # result (thresh = 0.2), eq hypocenter, station location, and hypocentral distance

mags = []
pgds = []
dists_km = []
dots = []

true_pos = []
false_neg = []

# Indices for rows with certain earthquake magnitudes 

logpgds_2175 = []
logpgds_17515 = []
logpgds_15125 = []
logpgds_1251 = []
logpgds_1075 = []
logpgds_07505 = []
logpgds_05025 = []
logpgds_0250 = []
logpgds_0025 = []
logpgds_02505 = []
logpgds_05075 = []
logpgds_0751 = []

for i in range(len(data)):
    
    if data[i][0] == 'nan':
        pass
    
    else:
        rupture = data[i][0]
        station = data[i][1]
        mag = float(data[i][2])
        pgd = float(data[i][3])
        result = data[i][4]
        hypodist_m = float(data[i][10])
        hypodist_km = hypodist_m/1000
        
        # print(pgd)
        
        if np.log10(pgd) <= -1.75:
            logpgds_2175.append(i)
        
        elif np.log10(pgd) > -1.75 and np.log10(pgd) <= -1.5:
            logpgds_17515.append(i)
            
        elif np.log10(pgd) > -1.5 and np.log10(pgd) <= -1.25:
            logpgds_15125.append(i)
        
        elif np.log10(pgd) > -1.25 and np.log10(pgd) <= -1:
            logpgds_1251.append(i)
        
        elif np.log10(pgd) > -1 and np.log10(pgd) <= -0.75:
            logpgds_1075.append(i)
        
        elif np.log10(pgd) > -0.75 and np.log10(pgd) <= -0.5:
            logpgds_07505.append(i)
            
        elif np.log10(pgd) > -0.5 and np.log10(pgd) <= -0.25:
            logpgds_05025.append(i)
        
        elif np.log10(pgd) > -0.25 and np.log10(pgd) <= 0:
            logpgds_0250.append(i)
        
        elif np.log10(pgd) > 0 and np.log10(pgd) <= 0.25:
            logpgds_0025.append(i)
            
        elif np.log10(pgd) > 0.25 and np.log10(pgd) <= 0.5:
            logpgds_02505.append(i)
        
        elif np.log10(pgd) > 0.5 and np.log10(pgd) <= 0.75:
            logpgds_05075.append(i)
            
        elif np.log10(pgd) > 0.75 and np.log10(pgd) <= 1:
            logpgds_0751.append(i)
        
        if result == 'true_pos': # predicted 1, target 1
            dot = 1
            mags.append(mag)
            pgds.append(pgd)
            dots.append(dot)
            dists_km.append(hypodist_km)
            true_pos.append(1)
        
        elif result == 'false_neg': # predicted 0, target 1
            dot = 0
            mags.append(mag)
            pgds.append(pgd)
            dots.append(dot)
            dists_km.append(hypodist_km)
            false_neg.append(1)
            
        else:
            pass
    
mags = np.asarray(mags)
pgds = np.asarray(pgds)
log_pgds = np.log10(pgds)
dists_km = np.asarray(dists_km)
dots = np.asarray(dots)

# print(pgds)

# print(min(log_pgds))

number_correct = len(true_pos)
number_incorrect = len(false_neg)
total_number = number_correct + number_incorrect

print('True positive: ' + str(number_correct))
print('False negative: ' + str(number_incorrect))
print('Total number: ' + str(total_number))
print('Accuracy for real earthquakes: ' + str(100*(number_correct/total_number)) + '%')

# plt.figure(dpi=100)
# plt.scatter(mags, dots, alpha = 0.5, s = 5, marker = 'o')
# plt.xlabel('Magnitude')
# plt.ylabel('1 = true positive, 0 = false negative')
# plt.title('Magnitude')

# plt.figure(dpi=100)
# plt.scatter(dists_km, dots, alpha = 0.5, s = 5, marker = 'o')
# plt.xlabel('Hypocentral distance (km)')
# plt.ylabel('1 = true positive, 0 = false negative')
# plt.title('Hypocentral distance (km)')

# plt.figure(dpi=100)
# plt.scatter(pgds, dots, alpha = 0.5, s = 5, marker = 'o')
# plt.xlabel('Peak ground displacement (m)')
# plt.ylabel('1 = true positive, 0 = false negative')
# plt.title('Peak ground displacement (m)')

# plt.figure(dpi=100)
# plt.scatter(log_pgds, dots, alpha = 0.5, s = 5, marker = 'o')
# plt.xlabel('Log of peak ground displacement (m)')
# plt.ylabel('1 = true positive, 0 = false negative')
# plt.title('Log of peak ground displacement (m)')

# plt.figure(dpi=500)
# plt.hist(log_pgds, bins=50)
# # plt.xlim(-6,0)
# plt.ylabel('Number of records in bin')
# plt.xlabel('Log peak ground displacement (m)')
# plt.title('Distribution of PGDs in data')
# plt.savefig('logpgd_histogram.png', format='PNG')
# plt.close()

np.savetxt('log_waveformpgds_txt.txt', log_pgds)

# # Making 0.05 PGD bins and calculating accuracies

# logpgds_2175 = []
# logpgds_17515 = []
# logpgds_15125 = []
# logpgds_1251 = []
# logpgds_1075 = []
# logpgds_07505 = []
# logpgds_05025 = []
# logpgds_0250 = []
# logpgds_0025 = []
# logpgds_02505 = []
# logpgds_05075 = []
# logpgds_0751 = []

print(len(logpgds_2175)+len(logpgds_17515)+len(logpgds_15125)+len(logpgds_1251)+len(logpgds_1075)+len(logpgds_07505)+len(logpgds_05025)+len(logpgds_0250)+len(logpgds_0025)+len(logpgds_02505)+len(logpgds_05075)+len(logpgds_0751))

# Calculating accuracies for bins

accuracy_percentages = []

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -2 to -1.75 m')
print('-------------------------------------------------------------')

truepos_2175 = []
falseneg_2175 = []

for index in logpgds_2175:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_2175.append(1)
    elif result == 'false_neg':
        falseneg_2175.append(1)
        
number_correct_2175 = len(truepos_2175)
number_incorrect_2175 = len(falseneg_2175)
total_number_2175 = number_correct_2175 + number_incorrect_2175

print('True positive: ' + str(number_correct_2175))
print('False negative: ' + str(number_incorrect_2175))
print('Total number: ' + str(total_number_2175))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_2175/total_number_2175)) + '%')

accuracy_2175 = 100*(number_correct_2175/total_number_2175)
accuracy_percentages.append(accuracy_2175)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -1.75 to -1.5 m')
print('-------------------------------------------------------------')

truepos_17515 = []
falseneg_17515 = []

for index in logpgds_17515:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_17515.append(1)
    elif result == 'false_neg':
        falseneg_17515.append(1)
        
number_correct_17515 = len(truepos_17515)
number_incorrect_17515 = len(falseneg_17515)
total_number_17515 = number_correct_17515 + number_incorrect_17515

print('True positive: ' + str(number_correct_17515))
print('False negative: ' + str(number_incorrect_17515))
print('Total number: ' + str(total_number_17515))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_17515/total_number_17515)) + '%')

accuracy_17515 = 100*(number_correct_17515/total_number_17515)
accuracy_percentages.append(accuracy_17515)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -1.5 to -1.25 m')
print('-------------------------------------------------------------')

truepos_15125 = []
falseneg_15125 = []

for index in logpgds_15125:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_15125.append(1)
    elif result == 'false_neg':
        falseneg_15125.append(1)
        
number_correct_15125 = len(truepos_15125)
number_incorrect_15125 = len(falseneg_15125)
total_number_15125 = number_correct_15125 + number_incorrect_15125

print('True positive: ' + str(number_correct_15125))
print('False negative: ' + str(number_incorrect_15125))
print('Total number: ' + str(total_number_15125))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_15125/total_number_15125)) + '%')

accuracy_15125 = 100*(number_correct_15125/total_number_15125)
accuracy_percentages.append(accuracy_15125)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -1.25 to -1 m')
print('-------------------------------------------------------------')

truepos_1251 = []
falseneg_1251 = []

for index in logpgds_1251:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_1251.append(1)
    elif result == 'false_neg':
        falseneg_1251.append(1)
        
number_correct_1251 = len(truepos_1251)
number_incorrect_1251 = len(falseneg_1251)
total_number_1251 = number_correct_1251 + number_incorrect_1251

print('True positive: ' + str(number_correct_1251))
print('False negative: ' + str(number_incorrect_1251))
print('Total number: ' + str(total_number_1251))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_1251/total_number_1251)) + '%')

accuracy_1251 = 100*(number_correct_1251/total_number_1251)
accuracy_percentages.append(accuracy_1251)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -1 to -0.75 m')
print('-------------------------------------------------------------')

truepos_1075 = []
falseneg_1075 = []

for index in logpgds_1075:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_1075.append(1)
    elif result == 'false_neg':
        falseneg_1075.append(1)
        
number_correct_1075 = len(truepos_1075)
number_incorrect_1075 = len(falseneg_1075)
total_number_1075 = number_correct_1075 + number_incorrect_1075

print('True positive: ' + str(number_correct_1075))
print('False negative: ' + str(number_incorrect_1075))
print('Total number: ' + str(total_number_1075))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_1075/total_number_1075)) + '%')

accuracy_1075 = 100*(number_correct_1075/total_number_1075)
accuracy_percentages.append(accuracy_1075)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -0.75 to -0.5 m')
print('-------------------------------------------------------------')

truepos_07505 = []
falseneg_07505 = []

for index in logpgds_07505:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_07505.append(1)
    elif result == 'false_neg':
        falseneg_07505.append(1)
        
number_correct_07505 = len(truepos_07505)
number_incorrect_07505 = len(falseneg_07505)
total_number_07505 = number_correct_07505 + number_incorrect_07505

print('True positive: ' + str(number_correct_07505))
print('False negative: ' + str(number_incorrect_07505))
print('Total number: ' + str(total_number_07505))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_07505/total_number_07505)) + '%')

accuracy_07505 = 100*(number_correct_07505/total_number_07505)
accuracy_percentages.append(accuracy_07505)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -0.5 to -0.25 m')
print('-------------------------------------------------------------')

truepos_05025 = []
falseneg_05025 = []

for index in logpgds_05025:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_05025.append(1)
    elif result == 'false_neg':
        falseneg_05025.append(1)
        
number_correct_05025 = len(truepos_05025)
number_incorrect_05025 = len(falseneg_05025)
total_number_05025 = number_correct_05025 + number_incorrect_05025

print('True positive: ' + str(number_correct_05025))
print('False negative: ' + str(number_incorrect_05025))
print('Total number: ' + str(total_number_05025))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_05025/total_number_05025)) + '%')

accuracy_05025 = 100*(number_correct_05025/total_number_05025)
accuracy_percentages.append(accuracy_05025)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD -0.25 to 0 m')
print('-------------------------------------------------------------')

truepos_0250 = []
falseneg_0250 = []

for index in logpgds_0250:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_0250.append(1)
    elif result == 'false_neg':
        falseneg_0250.append(1)
        
number_correct_0250 = len(truepos_0250)
number_incorrect_0250 = len(falseneg_0250)
total_number_0250 = number_correct_0250 + number_incorrect_0250

print('True positive: ' + str(number_correct_0250))
print('False negative: ' + str(number_incorrect_0250))
print('Total number: ' + str(total_number_0250))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_0250/total_number_0250)) + '%')

accuracy_0250 = 100*(number_correct_0250/total_number_0250)
accuracy_percentages.append(accuracy_0250)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD 0 to 0.25 m')
print('-------------------------------------------------------------')

truepos_0025 = []
falseneg_0025 = []

for index in logpgds_0025:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_0025.append(1)
    elif result == 'false_neg':
        falseneg_0025.append(1)
        
number_correct_0025 = len(truepos_0025)
number_incorrect_0025 = len(falseneg_0025)
total_number_0025 = number_correct_0025 + number_incorrect_0025

print('True positive: ' + str(number_correct_0025))
print('False negative: ' + str(number_incorrect_0025))
print('Total number: ' + str(total_number_0025))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_0025/total_number_0025)) + '%')

accuracy_0025 = 100*(number_correct_0025/total_number_0025)
accuracy_percentages.append(accuracy_0025)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD 0.25 to 0.5 m')
print('-------------------------------------------------------------')

truepos_02505 = []
falseneg_02505 = []

for index in logpgds_02505:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_02505.append(1)
    elif result == 'false_neg':
        falseneg_02505.append(1)
        
number_correct_02505 = len(truepos_02505)
number_incorrect_02505 = len(falseneg_02505)
total_number_02505 = number_correct_02505 + number_incorrect_02505

print('True positive: ' + str(number_correct_02505))
print('False negative: ' + str(number_incorrect_02505))
print('Total number: ' + str(total_number_02505))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_02505/total_number_02505)) + '%')

accuracy_02505 = 100*(number_correct_02505/total_number_02505)
accuracy_percentages.append(accuracy_02505)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD 0.5 to 0.75 m')
print('-------------------------------------------------------------')

truepos_05075 = []
falseneg_05075 = []

for index in logpgds_05075:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_05075.append(1)
    elif result == 'false_neg':
        falseneg_05075.append(1)
        
number_correct_05075 = len(truepos_05075)
number_incorrect_05075 = len(falseneg_05075)
total_number_05075 = number_correct_05075 + number_incorrect_05075

print('True positive: ' + str(number_correct_05075))
print('False negative: ' + str(number_incorrect_05075))
print('Total number: ' + str(total_number_05075))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_05075/total_number_05075)) + '%')

accuracy_05075 = 100*(number_correct_05075/total_number_05075)
accuracy_percentages.append(accuracy_05075)

print(' ')
print('-------------------------------------------------------------')
print('Log PGD 0.75 to 1 m')
print('-------------------------------------------------------------')

truepos_0751 = []
falseneg_0751 = []

for index in logpgds_0751:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_0751.append(1)
    elif result == 'false_neg':
        falseneg_0751.append(1)
        
number_correct_0751 = len(truepos_0751)
number_incorrect_0751 = len(falseneg_0751)
total_number_0751 = number_correct_0751 + number_incorrect_0751

print('True positive: ' + str(number_correct_0751))
print('False negative: ' + str(number_incorrect_0751))
print('Total number: ' + str(total_number_0751))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_0751/total_number_0751)) + '%')

accuracy_0751 = 100*(number_correct_0751/total_number_0751)
accuracy_percentages.append(accuracy_0751)

# Plotting accuracies by bin

bar_positions = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75]

print(len(bar_positions))
print(len(accuracy_percentages))

plt.figure(figsize = (9,5.25), dpi=300)
# plt.figure(figsize = (9,5))
plt.grid(axis = 'y', zorder = 0)
plt.bar(bar_positions, accuracy_percentages, width = 0.25, color = 'lightskyblue', align = 'edge', edgecolor = 'black', zorder = 3)
plt.axvline(x = np.log10(0.02), color = 'darkorange', linewidth = 3, zorder = 3)
plt.xlim(-2,1)
plt.ylim(0,100)
plt.ylabel('Accuracy (%)', fontsize = 16)
plt.xlabel('Log PGD (m)', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Algorithm accuracy (threshold = 0.615)', fontsize = 15)
plt.suptitle('Log peak ground placement (m)', fontsize = 17)

# plt.show()
plt.savefig('binned_logwaveformpgd_acc.png', format='PNG')
plt.close()












