#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:45:51 2022

@author: sydneydybing
"""
import numpy as np
import matplotlib.pyplot as plt

N_barpos = np.loadtxt('N_barpos.txt')
N_accper = np.loadtxt('N_accper.txt')
E_barpos = np.loadtxt('E_barpos.txt')
E_accper = np.loadtxt('E_accper.txt')
Z_barpos = np.loadtxt('Z_barpos.txt')
Z_accper = np.loadtxt('Z_accper.txt')

width = 0.12

plt.figure(figsize = (10,7), dpi=300)
# plt.figure(figsize = (9,5))
plt.grid(axis = 'y', zorder = 0)
plt.bar(N_barpos - width, N_accper, width, color = '#2DADB4', align = 'center', edgecolor = 'black', zorder = 3, label = 'N')
plt.bar(E_barpos, E_accper, width, color = '#001528', align = 'center', edgecolor = 'black', zorder = 3, label = 'E')
plt.bar(Z_barpos + width, Z_accper, width, color = '#E9072D', align = 'center', edgecolor = 'black', zorder = 3, label = 'Z')
# plt.axvline(x = np.log10(0.02), color = 'darkorange', linewidth = 3)
plt.xlim(-2,2.18)
plt.ylim(0,100)
plt.ylabel('Accuracy (%)', fontsize = 16)
plt.xlabel('Log SNR', fontsize = 16)
plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], fontsize = 14)
plt.yticks(fontsize = 14)
plt.title('Algorithm accuracy (threshold = 0.615)', fontsize = 15)
plt.suptitle('Log signal to noise ratio by component', fontsize = 19)
plt.legend(fontsize = 16)

# plt.show()
plt.savefig('binned_logacc_allcomps.png', format='PNG')
plt.close()
