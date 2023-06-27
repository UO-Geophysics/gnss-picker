#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:16:45 2021

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/sydneydybing/GNSS-CNN_repo/GNSS-CNN/shuffle_trainfull/analysis_w_hypdist_result.csv', dtype = str)

print(len(data))
# minmag = min(data[:,2])
# maxmag = max(data[:,2])
# minpgd = min(data[:,3])
# maxpgd = max(data[:,3])
# minhyp = min(data[:,10])
# maxhyp = max(data[:,10])

mags = []

for idx in range(len(data)):
    
    mag = data[idx,2]
    
    if mag != 'nan':
        
        mags.append(mag)

print(len(mags))
print(min(mags))
print(max(mags))


# This CSV has columns for rupture number, station, magnitude, pgd, classification 
# result (thresh = 0.2), eq hypocenter, station location, and hypocentral distance

mags = []
pgds = []
dists_km = []
dots = []

true_pos = []
false_neg = []

# Indices for rows with certain earthquake magnitudes 

mags_5756 = []
mags_6625 = []
mags_62565 = []
mags_65675 = []
mags_6757 = []
mags_7725 = []
mags_72575 = []

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
        
        if mag <= 6:
            mags_5756.append(i)
        
        elif mag > 6 and mag <= 6.25:
            mags_6625.append(i)
        
        elif mag > 6.25 and mag <= 6.5:
            mags_62565.append(i)
            
        elif mag > 6.5 and mag <= 6.75:
            mags_65675.append(i)
        
        elif mag > 6.75 and mag <= 7:
            mags_6757.append(i)
        
        elif mag > 7 and mag <= 7.25:
            mags_7725.append(i)
        
        elif mag > 7.25 and mag <= 7.5:
            mags_72575.append(i)
        
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

# mags_55575 = []
# mags_5756 = []
# mags_6625 = []
# mags_62565 = []
# mags_65675 = []
# mags_6757 = []
# mags_7725 = []
# mags_72575 = []

print(len(mags_5756)+len(mags_6625)+len(mags_62565)+len(mags_65675)+len(mags_6757)+len(mags_7725)+len(mags_72575))

# Calculating accuracies for bins

accuracy_percentages = []

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M5.75-6')
print('-------------------------------------------------------------')

truepos_5756 = []
falseneg_5756 = []

for index in mags_5756:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_5756.append(1)
    elif result == 'false_neg':
        falseneg_5756.append(1)
        
number_correct_5756 = len(truepos_5756)
number_incorrect_5756 = len(falseneg_5756)
total_number_5756 = number_correct_5756 + number_incorrect_5756

print('True positive: ' + str(number_correct_5756))
print('False negative: ' + str(number_incorrect_5756))
print('Total number: ' + str(total_number_5756))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_5756/total_number_5756)) + '%')

accuracy_5756 = 100*(number_correct_5756/total_number_5756)
accuracy_percentages.append(accuracy_5756)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M6-6.25')
print('-------------------------------------------------------------')

truepos_6625 = []
falseneg_6625 = []

for index in mags_6625:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_6625.append(1)
    elif result == 'false_neg':
        falseneg_6625.append(1)
        
number_correct_6625 = len(truepos_6625)
number_incorrect_6625 = len(falseneg_6625)
total_number_6625 = number_correct_6625 + number_incorrect_6625

print('True positive: ' + str(number_correct_6625))
print('False negative: ' + str(number_incorrect_6625))
print('Total number: ' + str(total_number_6625))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_6625/total_number_6625)) + '%')

accuracy_6625 = 100*(number_correct_6625/total_number_6625)
accuracy_percentages.append(accuracy_6625)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M6.25-6.5')
print('-------------------------------------------------------------')

truepos_62565 = []
falseneg_62565 = []

for index in mags_62565:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_62565.append(1)
    elif result == 'false_neg':
        falseneg_62565.append(1)
        
number_correct_62565 = len(truepos_62565)
number_incorrect_62565 = len(falseneg_62565)
total_number_62565 = number_correct_62565 + number_incorrect_62565

print('True positive: ' + str(number_correct_62565))
print('False negative: ' + str(number_incorrect_62565))
print('Total number: ' + str(total_number_62565))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_62565/total_number_62565)) + '%')

accuracy_62565 = 100*(number_correct_62565/total_number_62565)
accuracy_percentages.append(accuracy_62565)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M6.5-6.75')
print('-------------------------------------------------------------')

truepos_65675 = []
falseneg_65675 = []

for index in mags_65675:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_65675.append(1)
    elif result == 'false_neg':
        falseneg_65675.append(1)
        
number_correct_65675 = len(truepos_65675)
number_incorrect_65675 = len(falseneg_65675)
total_number_65675 = number_correct_65675 + number_incorrect_65675

print('True positive: ' + str(number_correct_65675))
print('False negative: ' + str(number_incorrect_65675))
print('Total number: ' + str(total_number_65675))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_65675/total_number_65675)) + '%')

accuracy_65675 = 100*(number_correct_65675/total_number_65675)
accuracy_percentages.append(accuracy_65675)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M6.75-7')
print('-------------------------------------------------------------')

truepos_6757 = []
falseneg_6757 = []

for index in mags_6757:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_6757.append(1)
    elif result == 'false_neg':
        falseneg_6757.append(1)
        
number_correct_6757 = len(truepos_6757)
number_incorrect_6757 = len(falseneg_6757)
total_number_6757 = number_correct_6757 + number_incorrect_6757

print('True positive: ' + str(number_correct_6757))
print('False negative: ' + str(number_incorrect_6757))
print('Total number: ' + str(total_number_6757))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_6757/total_number_6757)) + '%')

accuracy_6757 = 100*(number_correct_6757/total_number_6757)
accuracy_percentages.append(accuracy_6757)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M7-7.25')
print('-------------------------------------------------------------')

truepos_7725 = []
falseneg_7725 = []

for index in mags_7725:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_7725.append(1)
    elif result == 'false_neg':
        falseneg_7725.append(1)
        
number_correct_7725 = len(truepos_7725)
number_incorrect_7725 = len(falseneg_7725)
total_number_7725 = number_correct_7725 + number_incorrect_7725

print('True positive: ' + str(number_correct_7725))
print('False negative: ' + str(number_incorrect_7725))
print('Total number: ' + str(total_number_7725))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_7725/total_number_7725)) + '%')

accuracy_7725 = 100*(number_correct_7725/total_number_7725)
accuracy_percentages.append(accuracy_7725)

print(' ')
print('-------------------------------------------------------------')
print('Earthquakes M7.25-7.5')
print('-------------------------------------------------------------')

truepos_72575 = []
falseneg_72575 = []

for index in mags_72575:
    
    # print(index)
    event_row = data[index]
    # print(event_row)
    
    result = event_row[4]
    # print(result)
    
    if result == 'true_pos':
        truepos_72575.append(1)
    elif result == 'false_neg':
        falseneg_72575.append(1)
        
number_correct_72575 = len(truepos_72575)
number_incorrect_72575 = len(falseneg_72575)
total_number_72575 = number_correct_72575 + number_incorrect_72575

print('True positive: ' + str(number_correct_72575))
print('False negative: ' + str(number_incorrect_72575))
print('Total number: ' + str(total_number_72575))
print('Accuracy for real earthquakes: ' + str(100*(number_correct_72575/total_number_72575)) + '%')

accuracy_72575 = 100*(number_correct_72575/total_number_72575)
accuracy_percentages.append(accuracy_72575)

# Plotting accuracies for 0.25 bins

bar_positions = [5.75,6,6.25,6.5,6.75,7,7.25]

# print(bar_positions)
# print(accuracy_percentages)

plt.figure(figsize = (9,5.25), dpi = 300)
# plt.figure(figsize = (9,5))
plt.grid(axis = 'y', zorder = 0)
plt.bar(bar_positions, accuracy_percentages, width = 0.5, color = 'lightskyblue', align = 'edge', edgecolor = 'black', zorder = 3)
plt.xlim(5.75,7.5)
plt.ylim(0,100)
plt.xticks([5.75,6,6.25,6.5,6.75,7,7.25,7.5])
plt.ylabel('Accuracy (%)', fontsize = 16)
plt.xlabel('Magnitude', fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.suptitle('Magnitude', fontsize = 17)
plt.title('Algorithm accuracy (threshold = 0.615)', fontsize = 15)

# plt.show()
plt.savefig('binned_mag_acc.png', format='PNG')
# plt.savefig('NEW_binned_mag_acc.png', format='PNG')
plt.close()

    
    











    
