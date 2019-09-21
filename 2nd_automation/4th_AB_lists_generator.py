import numpy as np
import time
import matplotlib.pyplot as plt
import os, glob
import gc
import sys
sys.path.insert(0, '/home/sonic/Coding/Git/CPMG_Analysis/2nd_automation/')
from decom_utils import *

TIME_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]

MAGNETIC_FIELD = 403.7139663551402            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000               # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32
SAVE_ABLIST_DIR = '/home/sonic/Coding/Git/ABlists_4th_model/'
margin = 0
width_threshold = 70


# AB의 전체 후보군 리스트
A_search_min = -100250
A_search_max = 100250          # Set the boundary between -100000 ~ 100000 (Hz) for A
B_search_min = 500
B_search_max = 80500           # Set the boundary between 0 ~ 200000 (Hz) for B
A_steps = 500                  # interval between A points
B_steps = 1000                 # interval between B points

n_A_samples = (A_search_max - A_search_min) // A_steps
n_B_samples = (B_search_max - B_search_min) // B_steps

TOTAL_AB_ARRAY = np.zeros((n_A_samples * n_B_samples, 2))

B_FIXED_VALUE = B_search_min
for i in range(n_B_samples):
    test_array = np.array([np.arange(A_search_min, A_search_max, A_steps), np.full((n_A_samples,), B_FIXED_VALUE)])
    test_array = test_array.transpose()
    test_array = np.round(test_array,2)
    TOTAL_AB_ARRAY[i*n_A_samples:(i+1)*n_A_samples] = test_array
    B_FIXED_VALUE += B_steps
TOTAL_AB_ARRAY.shape

# 모델을 만들 AB 범위
A_start = -60000
A_end = 60000
A_step = 500
B_start = 2000
B_end = 70000
B_step = 1000

A_interval = 1000
B_interval = 2000

A_pair1 = np.arange(A_start, A_end, A_interval)
A_pair2 = np.arange(A_start, A_end, A_interval) + A_interval
A_range_list = [[a,b] for a,b in zip(A_pair1, A_pair2)]

B_pair1 = np.arange(B_start, B_end, B_interval)
B_pair2 = np.arange(B_start, B_end, B_interval) + B_interval
B_range_list = [[a,b] for a,b in zip(B_pair1, B_pair2)]

AB_dic = {}
count = 0
for idx1, A_range in enumerate(A_range_list):
    for idx2, B_range in enumerate(B_range_list):
        AB_dic[count] = [[idx1, idx2], [A_range, B_range]]
        count += 1

# 뽑을 spin 개수 및 target 스핀 한 개당 만들 data의 개수
n_of_spins = 39
n_of_small_spins = 20 
n_of_samples_per_target = 15000

# AB 후보 범위
A_candidate_max = 70000
A_candidate_min = -70000
B_candidate_max = 70000
B_candidate_min = 2000

# small AB 후보 범위
A_small_max = 25000
A_small_min = -25000
B_small_max = 14000
B_small_min = 2000

target_ABs = []
AB_range_list = []
total_fil_idx = []

for idx in AB_dic:
    tic = time.time()
    [[A_idx, B_idx], [A_range, B_range]] = AB_dic[idx]
    A_min = A_range[0]
    A_max = A_range[1]
    B_min = B_range[0]
    B_max = B_range[1]

    AB_range = 'A(' + str(A_min//1000) +'-'+ str(A_max//1000) + ')_B(' + str(B_min//1000) +'-'+ str(B_max//1000) + ')'

    args = TOTAL_AB_ARRAY, A_candidate_max, A_candidate_min, B_candidate_max, B_candidate_min, A_small_max, \
           A_small_min, B_small_max, B_small_min, A_max, A_min, B_max, B_min, A_step, B_step, B_idx

    target_array, AB_candidate_array, small_AB_array = gen_AB_candidates_arrays(*args)
    target_ABs.append(target_array)
    AB_range_list.append([A_range, B_range])
    if (idx+1)%3 == 0:
        target_ABs = np.array(target_ABs)
        total_ABlists = gen_ABlists(target_ABs, AB_candidate_array, small_AB_array, n_of_spins, n_of_small_spins, n_of_samples_per_target)
        np.save(SAVE_ABLIST_DIR + "total_ABlists_{}_{}_{}_{}_{}.npy".format((idx+1)//3, AB_range_list[0][0][0], AB_range_list[0][0][1], AB_range_list[0][1][0], AB_range_list[2][1][1]), total_ABlists)
        
        for AB in AB_range_list: 
            A, B = AB[0][0] + 500, AB[1][0] + 1000 
            M_value = M_list_return(time_data, WL_VALUE, [[A*2*np.pi,B*2*np.pi]], N_PULSE_32)
            filtered_idx = get_filtered_idx(M_value, margin=margin, width_threshold=width_threshold)
            total_fil_idx.append(filtered_idx)
        filtered_indices = list(set(list(total_fil_idx[0])+list(total_fil_idx[1])+list(total_fil_idx[2])))
        np.save(SAVE_ABLIST_DIR + "total_ABlists_{}_filtered_idx.npy".format((idx+1)//3), np.array(filtered_indices))

        print("==========================")
        print("Generated Range\n", AB_range_list)

        target_ABs = [] 
        AB_range_list = []
        total_fil_idx = []