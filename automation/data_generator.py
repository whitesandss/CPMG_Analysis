import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from decom_utils.py import *
import numpy as np
import time
import os
import pickle

time_table = np.load('/home/sonic/Coding/Decomposition/TimTaminiau/20190610_time_no_duplicated.npy')[:12000]*1e-6

MAGNETIC_FIELD = 403            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000 # Unit: Herts
WL_VALUE = np.round(MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi, 3)
N_PULSE_32 = 32
SAVE_AB_LISTS_DIR = '/home/sonic/Coding/Decomposition/Simul_Data/AB_lists/'

A_search_min = -100000
A_search_max = 100000          # Set the boundary between -100000 ~ 100000 (Hz) for A
B_search_min = 0
B_search_max = 80000           # Set the boundary between 0 ~ 200000 (Hz) for B
A_steps = 1000
B_steps = 1000

n_of_samples_per_target = 20000
n_of_spins = 7

n_A_samples = (A_search_max - A_search_min) // A_steps
n_B_samples = (B_search_max - B_search_min) // B_steps

TOTAL_TEST_ARRAY = np.zeros((n_A_samples * n_B_samples, 2))

B_FIXED_VALUE = B_search_min
for i in range(n_B_samples):
    test_array = np.array([np.arange(A_search_min, A_search_max, A_steps), np.full((n_A_samples,), B_FIXED_VALUE)])
    test_array = test_array.transpose()
    test_array = np.round(test_array,2)
    TOTAL_TEST_ARRAY[i*n_A_samples:(i+1)*n_A_samples] = test_array
    B_FIXED_VALUE += B_steps
TOTAL_TEST_ARRAY.shape

A_start = -70000
A_end = 70000
A_step = 4000
B_start = 16000
B_end = 70000
B_step = 4000

end_to_end_range = 6000

A_pair1 = np.arange(A_start, A_end, A_step)
A_pair2 = np.arange(A_start, A_end+A_step, A_step) + end_to_end_range
A_range_list = [[a,b] for a,b in zip(A_pair1, A_pair2)]

B_pair1 = np.arange(B_start, B_end, B_step)
B_pair2 = np.arange(B_start, B_end, B_step) + end_to_end_range
B_range_list = [[a,b] for a,b in zip(B_pair1, B_pair2)]



def gen_AB_lists_to_px_lists(start_idx, target_AB_included, none_target_AB, count):
    n_samples = 1400
    # start_n_samples = n_samples*25*(count-1)
    total_target_M_list = np.zeros((n_samples, len(time_table)))
    total_none_M_list = np.zeros((n_samples, len(time_table)))
    print("start_idx: ", start_idx)
    for i in range(n_samples):
        total_target_M_list[i] = M_list_return(time_table, WL_VALUE, target_AB_included[start_idx+i], N_PULSE_32)
        total_none_M_list[i] = M_list_return(time_table, WL_VALUE, none_target_AB[start_idx+i], N_PULSE_32)
    print("count: ", count, end='\r')
    return total_target_M_list, total_none_M_list

def generate(A_range, A_idx, B_range, B_idx, N_files = 5, PoolProcess=23):
    A_min = A_range[0]
    A_max = A_range[1]
    B_min = B_range[0]
    B_max = B_range[1]
    
    target_index = str(A_idx) + '-' + str(B_idx)
    
    AB_range = 'A(' + str(A_min//1000) +'-'+ str(A_max//1000) + ')_B(' + str(B_min//1000) +'-'+ str(B_max//1000) + ')'

    A_candidate_max = 70000
    A_candidate_min = -70000
    B_candidate_max = 70000
    B_candidate_min = 5000
    
    indices = np.where((TOTAL_TEST_ARRAY[:,0]>=A_min) & (TOTAL_TEST_ARRAY[:,0]<=A_max) & (TOTAL_TEST_ARRAY[:,1]>=B_min) & (TOTAL_TEST_ARRAY[:,1]<=B_max))
    target_array = TOTAL_TEST_ARRAY[indices]

    candidate_boolen_index = ((TOTAL_TEST_ARRAY[:,1]>=B_candidate_min) & (TOTAL_TEST_ARRAY[:,1]<=B_candidate_max))   \
                            & ((TOTAL_TEST_ARRAY[:,0]>=A_candidate_min) & (TOTAL_TEST_ARRAY[:,0]<=A_candidate_max)) \
                            & (((TOTAL_TEST_ARRAY[:,1]<=(B_min-5000)) | (TOTAL_TEST_ARRAY[:,1]>=(B_max+5000)))      \
                            | ((TOTAL_TEST_ARRAY[:,0]<=(A_min-10000)) | (TOTAL_TEST_ARRAY[:,0]>=(A_max+10000))))
    AB_candidate_array = TOTAL_TEST_ARRAY[candidate_boolen_index]
    
    AB_candidate_array = AB_candidate_array * 2 * np.pi
    target_array = target_array * 2 * np.pi

    candidate_idx = np.arange(len(AB_candidate_array))
    n_of_targets = len(target_array)
    
    tic = time.time()
    target_AB_included = np.zeros((n_of_targets, n_of_samples_per_target, n_of_spins, 2))
    none_target_AB = np.zeros((n_of_targets, n_of_samples_per_target, n_of_spins, 2))
    target_temp = np.zeros((n_of_spins, 2))
    non_target_temp = np.zeros((n_of_spins, 2))
    
    for i in range(len(target_array)):
        target_temp = np.zeros((n_of_spins, 2))
        target_temp[0] = target_array[i]
        for j in range(n_of_samples_per_target):
            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-1)
            while len(set(indices_candi)) != n_of_spins-1:
                indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-1)
            target_temp[1:] = AB_candidate_array[indices_candi]
            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins)
            while len(set(indices_candi)) != n_of_spins:
                indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins)
            non_target_temp = AB_candidate_array[indices_candi]
            target_AB_included[i, j, :, :] = target_temp
            none_target_AB[i, j, :, :] = non_target_temp
    toc = time.time()
    
    print("Generate AB_lists Completed. : ", toc - tic)
    
    target_AB_included = target_AB_included.reshape(n_of_targets*n_of_samples_per_target, n_of_spins, 2)
    none_target_AB = none_target_AB.reshape(n_of_targets*n_of_samples_per_target, n_of_spins, 2)
    np.random.shuffle(target_AB_included)
    
    plt.figure(facecolor='w', figsize=(20,10))
    plt.plot(none_target_AB[:1000,0,0]/2/np.pi, none_target_AB[:1000,0,1]/2/np.pi, 'o')
    plt.plot(target_AB_included[:1000,0,0]/2/np.pi, target_AB_included[:1000,0,1]/2/np.pi, 'o')
    plt.savefig(SAVE_AB_LISTS_DIR + '{}_AB_distributions_'.format(target_index) + AB_range + '_'+ str(n_of_spins) + 'spin_ABlist.jpg')

    print("AB_range", AB_range)
    print('A_min: ', np.min(target_AB_included[:,0,0])/2/np.pi)
    print('A_max: ', np.max(target_AB_included[:,0,0])/2/np.pi)
    print('B_min: ', np.min(target_AB_included[:,0,1])/2/np.pi)
    print('B_max: ', np.max(target_AB_included[:,0,1])/2/np.pi)
    print("==============================")
    print('A_min(None): ', np.min(none_target_AB[:,0,0])/2/np.pi)
    print('A_max(None): ', np.max(none_target_AB[:,0,0])/2/np.pi)
    print('B_min(None): ', np.min(none_target_AB[:,0,1])/2/np.pi)
    print('B_max(None): ', np.max(none_target_AB[:,0,1])/2/np.pi)

    np.save(SAVE_AB_LISTS_DIR + '{}_none_target_'.format(target_index) + AB_range + '_'+ str(n_of_spins) + 'spin_ABlist.npy', none_target_AB)
    np.save(SAVE_AB_LISTS_DIR + '{}_target_'.format(target_index) + AB_range + '_' + str(n_of_spins) + 'spin_ABlist.npy', target_AB_included)        
    
    ###################################### AB_generated.

    SAVE_DATA_DIR = '/media/sonic/Samsung_T5/CPMG_data/' + target_index + '/'
    if not(os.path.isdir(SAVE_DATA_DIR)):
        os.mkdir(SAVE_DATA_DIR)

    for count_idx in range(1, N_files + 1):
        count = count_idx # 1씩 증가

        init_n_samples = 1400

        start_n_samples = init_n_samples*25*(count-1)
        # 날짜를 잘 로딩할 것
        none_target_AB = np.load(SAVE_AB_LISTS_DIR + '{}_none_target_'.format(target_index) + AB_range + '_'+ str(n_of_spins) + 'spin_ABlist.npy')
        target_AB_included = np.load(SAVE_AB_LISTS_DIR + '{}_target_'.format(target_index) + AB_range + '_'+ str(n_of_spins) + 'spin_ABlist.npy')
        n_spins = len(target_AB_included[0])

        pool = Pool(processes=PoolProcess)

        result1 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*0), target_AB_included, none_target_AB, count])
        result2 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*1), target_AB_included, none_target_AB, count])
        result3 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*2), target_AB_included, none_target_AB, count])
        result4 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*3), target_AB_included, none_target_AB, count])
        result5 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*4), target_AB_included, none_target_AB, count])
        result6 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*5), target_AB_included, none_target_AB, count])
        result7 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*6), target_AB_included, none_target_AB, count])
        result8 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*7), target_AB_included, none_target_AB, count])
        result9 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*8), target_AB_included, none_target_AB, count])
        result10 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*9), target_AB_included, none_target_AB, count])
        result11 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*10), target_AB_included, none_target_AB, count])
        result12 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*11), target_AB_included, none_target_AB, count])
        result13 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*12), target_AB_included, none_target_AB, count])
        result14 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*13), target_AB_included, none_target_AB, count])
        result15 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*14), target_AB_included, none_target_AB, count])
        result16 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*15), target_AB_included, none_target_AB, count])
        result17 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*16), target_AB_included, none_target_AB, count])
        result18 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*17), target_AB_included, none_target_AB, count])
        result19 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*18), target_AB_included, none_target_AB, count])
        result20 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*19), target_AB_included, none_target_AB, count])
        result21 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*20), target_AB_included, none_target_AB, count])
        result22 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*21), target_AB_included, none_target_AB, count])
        result23 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*22), target_AB_included, none_target_AB, count])
        if count==1:
            result24 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*23), target_AB_included, none_target_AB, count])
            result25 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*24), target_AB_included, none_target_AB, count])
        
        tic = time.time()
        total_target_px_list = np.zeros(((init_n_samples*46, len(time_table))))
        total_target_valid = np.zeros(((init_n_samples*2, len(time_table))))
        total_target_eval = np.zeros(((init_n_samples*2, len(time_table))))

        total_target_px_list[init_n_samples*0:init_n_samples*1],   total_target_px_list[init_n_samples*23:init_n_samples*24] = result1.get(timeout=None)
        total_target_px_list[init_n_samples*1:init_n_samples*2],   total_target_px_list[init_n_samples*24:init_n_samples*25] = result2.get(timeout=None)
        total_target_px_list[init_n_samples*2:init_n_samples*3],   total_target_px_list[init_n_samples*25:init_n_samples*26] = result3.get(timeout=None)
        total_target_px_list[init_n_samples*3:init_n_samples*4],   total_target_px_list[init_n_samples*26:init_n_samples*27] = result4.get(timeout=None)
        total_target_px_list[init_n_samples*4:init_n_samples*5],   total_target_px_list[init_n_samples*27:init_n_samples*28] = result5.get(timeout=None)
        total_target_px_list[init_n_samples*5:init_n_samples*6],   total_target_px_list[init_n_samples*28:init_n_samples*29] = result6.get(timeout=None)
        total_target_px_list[init_n_samples*6:init_n_samples*7],   total_target_px_list[init_n_samples*29:init_n_samples*30] = result7.get(timeout=None)
        total_target_px_list[init_n_samples*7:init_n_samples*8],   total_target_px_list[init_n_samples*30:init_n_samples*31] = result8.get(timeout=None)
        total_target_px_list[init_n_samples*8:init_n_samples*9],   total_target_px_list[init_n_samples*31:init_n_samples*32] = result9.get(timeout=None)
        total_target_px_list[init_n_samples*9:init_n_samples*10],  total_target_px_list[init_n_samples*32:init_n_samples*33] = result10.get(timeout=None)
        total_target_px_list[init_n_samples*10:init_n_samples*11], total_target_px_list[init_n_samples*33:init_n_samples*34] = result11.get(timeout=None)
        total_target_px_list[init_n_samples*11:init_n_samples*12],   total_target_px_list[init_n_samples*34:init_n_samples*35] = result12.get(timeout=None)
        total_target_px_list[init_n_samples*12:init_n_samples*13],   total_target_px_list[init_n_samples*35:init_n_samples*36] = result13.get(timeout=None)
        total_target_px_list[init_n_samples*13:init_n_samples*14],   total_target_px_list[init_n_samples*36:init_n_samples*37] = result14.get(timeout=None)
        total_target_px_list[init_n_samples*14:init_n_samples*15],   total_target_px_list[init_n_samples*37:init_n_samples*38] = result15.get(timeout=None)
        total_target_px_list[init_n_samples*15:init_n_samples*16],   total_target_px_list[init_n_samples*38:init_n_samples*39] = result16.get(timeout=None)
        total_target_px_list[init_n_samples*16:init_n_samples*17],   total_target_px_list[init_n_samples*39:init_n_samples*40] = result17.get(timeout=None)
        total_target_px_list[init_n_samples*17:init_n_samples*18],   total_target_px_list[init_n_samples*40:init_n_samples*41] = result18.get(timeout=None)
        total_target_px_list[init_n_samples*18:init_n_samples*19],   total_target_px_list[init_n_samples*41:init_n_samples*42] = result19.get(timeout=None)
        total_target_px_list[init_n_samples*19:init_n_samples*20],   total_target_px_list[init_n_samples*42:init_n_samples*43] = result20.get(timeout=None)
        total_target_px_list[init_n_samples*20:init_n_samples*21],   total_target_px_list[init_n_samples*43:init_n_samples*44] = result21.get(timeout=None)
        total_target_px_list[init_n_samples*21:init_n_samples*22],   total_target_px_list[init_n_samples*44:init_n_samples*45] = result22.get(timeout=None)
        total_target_px_list[init_n_samples*22:init_n_samples*23],   total_target_px_list[init_n_samples*45:init_n_samples*46] = result23.get(timeout=None)
        
        if count==1:
            total_target_valid[init_n_samples*0:init_n_samples*1], total_target_valid[init_n_samples*1:init_n_samples*2] = result24.get(timeout=None)
            total_target_eval[init_n_samples*0:init_n_samples*1], total_target_eval[init_n_samples*1:init_n_samples*2] = result25.get(timeout=None)
            
        toc = time.time()
        pool.close()
        pool.join()
        print("Calculated Time : ", toc-tic)

        ## train 파일 하나당 용량 
        ## init_n_samples=1024 인 경우 : 3.1GB
        np.save(SAVE_DATA_DIR + '{}_train_px_list_'.format(target_index) + AB_range + '_{}spin_{}.npy'.format(n_spins, count), total_target_px_list)
        del total_target_px_list
        print("Calculated Time : ", toc-tic)
        if count==1:
            np.save(SAVE_DATA_DIR + '{}_valid_px_list_'.format(target_index) + AB_range + '_{}spin.npy'.format(n_spins), total_target_valid)
            del total_target_valid
            np.save(SAVE_DATA_DIR + '{}_eval_px_list_'.format(target_index) + AB_range + '_{}spin.npy'.format(n_spins), total_target_eval)
            del total_target_eval

        print(AB_range+"Save Completed!, Count: ", count)        

if __name__ == "__main__":
    '''
    for A_idx, A_range in enumerate(A_range_list):
        for B_idx, B_range in enumerate(B_range_list):
    '''
    generate(A_range_list[0], 0, B_range_list[0], 0)