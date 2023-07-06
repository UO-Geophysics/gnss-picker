#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:40:02 2021

@author: sydneydybing
"""
import numpy as np
from ll_random_noise_picker_Tunguska import noise_picker
import multiprocessing as mp

ncpus = 50
# ncpus = 4
num_samples = 2668800
# num_samples = 20000
samples_per_cpu = num_samples / ncpus


stas = np.genfromtxt('/home/sdybing/GNSS/GNSS_stas.txt', usecols=[2], dtype=str)
dates = np.genfromtxt('/home/sdybing/GNSS/GNSS_dates.txt', dtype=str)

noise_data_path = '/hdd/Ridgecrest/mseeds/'
write_sample_path = '/home/sdybing/GNSS/noise_samples/mseeds_ll_2/'
# write_sample_path = '/home/sdybing/GNSS/noise_samples/'
save_npy_path = '/home/sdybing/GNSS/noise_samples/npys_ll_2/'
# save_npy_path = '/home/sdybing/GNSS/noise_samples/'
save_npy_name = 'noise_samples_2.npy'
progress_report_path = '/home/sdybing/GNSS/noise_samples/prog_rep_ll_2/'
# progress_report_path = '/home/sdybing/GNSS/noise_samples/'

if __name__ ==  '__main__':

    jobs = []
    for cpu_number in range(ncpus):
        p = mp.Process(target = noise_picker, args = (stas, dates, samples_per_cpu, noise_data_path, write_sample_path, cpu_number, save_npy_path, save_npy_name, progress_report_path))
        jobs.append(p)
        p.start()
