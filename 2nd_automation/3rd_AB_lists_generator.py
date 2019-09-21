import numpy as np
import time
import matplotlib.pyplot as plt
import os
import gc

time_table = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')[:12000]*1e-6

MAGNETIC_FIELD = 403.7139663551402    # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000       # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32
SAVE_AB_LISTS_DIR = '/data2/AB_list_3rd/'

# AB의 전체 후보군 리스트
A_search_min = -100000
A_search_max = 100000          # Set the boundary between -100000 ~ 100000 (Hz) for A
B_search_min = 0
B_search_max = 80000           # Set the boundary between 0 ~ 200000 (Hz) for B
A_steps = 500                  # interval between A points
B_steps = 1000                 # interval between B points

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

# 모델을 만들 AB 범위
A_start = -60000
A_end = 60000
A_step = 500
B_start = 2000
B_end = 70000
B_step = 1000

end_to_end_A_range = 1000
end_to_end_B_range = 2000

A_pair1 = np.arange(A_start, A_end, end_to_end_A_range)
A_pair2 = np.arange(A_start, A_end, end_to_end_A_range) + end_to_end_A_range
A_range_list = [[a,b] for a,b in zip(A_pair1, A_pair2)]

B_pair1 = np.arange(B_start, B_end, end_to_end_B_range)
B_pair2 = np.arange(B_start, B_end, end_to_end_B_range) + end_to_end_B_range
B_range_list = [[a,b] for a,b in zip(B_pair1, B_pair2)]

AB_dic = {}
count = 0
for idx1, A_range in enumerate(A_range_list):
    for idx2, B_range in enumerate(B_range_list):
        AB_dic[count] = [[idx1, idx2], [A_range, B_range]]
        count += 1

# 원하는 부분만 AB_list를 만들고 싶을때
heatmap_n_rows = len(B_range_list)
target_indices = []
A_index_min = 21  # 원하는 A영역
A_index_max = 26  
B_index_min = 9   # 원하는 B영역
B_index_max = 16

for i in range(len(AB_dic)):
    if (i//heatmap_n_rows > A_index_min) & (i//heatmap_n_rows < A_index_max):
        if (i%heatmap_n_rows > B_index_min) & (i%heatmap_n_rows < B_index_max):
            target_indices.append(i)

# 뽑을 spin 개수 및 target 스핀 한 개당 만들 data의 개수
n_of_spins=39
n_of_small_spins=20 
n_of_samples_per_target=30000

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

for idx in target_indices:
    [[idx1,idx2], [A_range, B_range]] = AB_dic[idx]
    
    A_min = A_range[0]
    A_max = A_range[1]
    B_min = B_range[0]
    B_max = B_range[1]

    AB_range = 'A(' + str(A_min//1000) +'-'+ str(A_max//1000) + ')_B(' + str(B_min//1000) +'-'+ str(B_max//1000) + ')'

    indices = np.where((TOTAL_TEST_ARRAY[:,0]>=A_min) & (TOTAL_TEST_ARRAY[:,0]<=A_max) & (TOTAL_TEST_ARRAY[:,1]>=B_min) & (TOTAL_TEST_ARRAY[:,1]<=B_max))
    target_array = TOTAL_TEST_ARRAY[indices]

    candidate_boolen_index = ((TOTAL_TEST_ARRAY[:,1]>=B_candidate_min) & (TOTAL_TEST_ARRAY[:,1]<=B_candidate_max))   \
                            & ((TOTAL_TEST_ARRAY[:,0]>=A_candidate_min) & (TOTAL_TEST_ARRAY[:,0]<=A_candidate_max)) \
                            & (((TOTAL_TEST_ARRAY[:,1]<=(B_min-B_step)) | (TOTAL_TEST_ARRAY[:,1]>=(B_max+B_step)))      \
                            | ((TOTAL_TEST_ARRAY[:,0]<=(A_min-A_step)) | (TOTAL_TEST_ARRAY[:,0]>=(A_max+A_step))))
    AB_candidate_array = TOTAL_TEST_ARRAY[candidate_boolen_index]

    small_AB_boolen_index = ((TOTAL_TEST_ARRAY[:,1]>=B_small_min) & (TOTAL_TEST_ARRAY[:,1]<=B_small_max))   \
                             & ((TOTAL_TEST_ARRAY[:,0]>=A_small_min) & (TOTAL_TEST_ARRAY[:,0]<=A_small_max)) \
                             & (((TOTAL_TEST_ARRAY[:,1]<=(B_min-B_step)) | (TOTAL_TEST_ARRAY[:,1]>=(B_max+B_step)))      \
                             | ((TOTAL_TEST_ARRAY[:,0]<=(A_min-A_step)) | (TOTAL_TEST_ARRAY[:,0]>=(A_max+A_step))))
    small_AB_array = TOTAL_TEST_ARRAY[small_AB_boolen_index]

    AB_candidate_array *= (2 * np.pi)
    target_array *= (2 * np.pi)
    small_AB_array *= (2 * np.pi)

    candidate_idx = np.arange(len(AB_candidate_array))
    n_of_targets = len(target_array)

    target_AB_included = np.zeros((n_of_targets, n_of_samples_per_target, n_of_spins, 2))
    none_target_AB = np.zeros((n_of_targets, n_of_samples_per_target, n_of_spins, 2))
    target_temp = np.zeros((n_of_spins-n_of_small_spins, 2))
    non_target_temp = np.zeros((n_of_spins-n_of_small_spins, 2))
    small_AB_temp = np.zeros((n_of_small_spins, 2))

    tic = time.time()
    for i in range(len(target_array)):
        target_temp[0] = target_array[i]
        for j in range(n_of_samples_per_target):

            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-1)
            while len(set(indices_candi)) != n_of_spins-n_of_small_spins-1:
                indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-1)
            target_temp[1:n_of_spins-n_of_small_spins] = AB_candidate_array[indices_candi]

            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins)
            while len(set(indices_candi)) != n_of_spins-n_of_small_spins:
                indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins)
            non_target_temp = AB_candidate_array[indices_candi]

            indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
            while len(set(indices_candi)) != n_of_small_spins:
                indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
            small_AB_temp = small_AB_array[indices_candi]

            target_AB_included[i, j, :n_of_spins-n_of_small_spins, :] = target_temp
            none_target_AB[i, j, :n_of_spins-n_of_small_spins, :] = non_target_temp

            target_AB_included[i, j, n_of_spins-n_of_small_spins:, :] = small_AB_temp
            none_target_AB[i, j, n_of_spins-n_of_small_spins:, :] = small_AB_temp
        print(i, j, end='\r')
    toc = time.time()

    print(AB_range+"Generation Completed : {} s".format(round(toc - tic)))

    target_AB_included = target_AB_included.reshape(n_of_targets*n_of_samples_per_target, n_of_spins, 2)
    none_target_AB = none_target_AB.reshape(n_of_targets*n_of_samples_per_target, n_of_spins, 2)
    np.random.shuffle(target_AB_included)

    plt.figure(facecolor='w', figsize=(20,10))
    plt.plot(none_target_AB[:1000,0,0]/2/np.pi, none_target_AB[:1000,0,1]/2/np.pi, 'o')
    plt.plot(target_AB_included[:1000,0,0]/2/np.pi, target_AB_included[:1000,0,1]/2/np.pi, 'o')
    plt.savefig(SAVE_AB_LISTS_DIR + "{}_AB_distributions_".format(idx) + AB_range + '_'+ str(n_of_spins) + 'spins.jpg')

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

    np.save(SAVE_AB_LISTS_DIR + '{}_none_target_'.format(idx) + AB_range + '_'+ str(n_of_spins) + 'spin_ABlist.npy', none_target_AB)
    np.save(SAVE_AB_LISTS_DIR + '{}_target_'.format(idx) + AB_range + '_' + str(n_of_spins) + 'spin_ABlist.npy', target_AB_included)
    del target_AB_included, none_target_AB
    gc.collect()