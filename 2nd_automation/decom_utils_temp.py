import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
import gc
from keras import backend as K
import time
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# DFT_list = pd.read_csv('./DFT_ABlist2.csv', header=None)
# DFT_list = DFT_list.drop_duplicates([0,1], keep='first')

# DFT_array = np.array(DFT_list)
# indices = np.argsort(DFT_array[:, 0])
# DFT_array = DFT_array[indices]

def time_table_gen(start, step, n_value):
    final_value = start + n_value * step
    time_table = np.arange(start, final_value, step)
    return time_table
    # M값을 구하는 공식에서 시간(t)를 return

def gaussian_slope_px(M_lists, time_table, time_index, px_mean_value_at_time):

    m_value_at_time = (px_mean_value_at_time * 2) - 1
    Gaussian_co = -time_table[time_index] / np.log(m_value_at_time)

    slope = np.exp(-time_table / Gaussian_co)
    slope = slope.reshape(1, len(time_table))
    M_lists_slope = M_lists * slope
    px_lists_slope = (M_lists_slope + 1) / 2
    return px_lists_slope, slope
    
def gaussian_slope(time_table, time_index, px_mean_value_at_time, M_list):
    m_value_at_time = (px_mean_value_at_time * 2) - 1
    Gaussian_co = -time_table[time_index] / np.log(m_value_at_time)

    slope = np.exp(-time_table / Gaussian_co)
    if M_list != None:
        M_list_slope = M_list * slope
        px_list_slope = (M_list_slope + 1) / 2
        return px_list_slope, slope
    return slope
    # 원래는 spin함수의 양자정보는 시간에 따라 exponetial 하게 decoherence함.
    # 그 감소하는 경향성을 return하는 것인데, taminiau 실험치는 거의 decoherence가 없어서 안씀.

# def M_list_return(time_table, wL_value, AB_list, n_pulse):

#     M_list = np.ones([len(time_table)])
#     for [A, B] in AB_list:                            ## 이 부분은 매우 중요. 2pi가 곱해져야 제대로 나옴. 중복계산도 고려해봐야함!!
        
#         w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)
#         mz = (A + wL_value) / w_tilda
#         mx = B / w_tilda

#         alpha = w_tilda * time_table
#         beta = wL_value * time_table

#         phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))
#         K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))
#         K2 = 1 + np.cos(phi)
#         K = pow(mx,2) * (K1 / K2)
#         M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)
#         M_list *= M_list_temp

#     return M_list

# def M_list_return_bulk(time_table, wL_value, AB_many_list, n_pulse):
    
#     n_samples = len(AB_many_list)
#     n_spins = len(AB_many_list[0])
#     n_time_values = len(time_table)
#     AB_list = AB_many_list.reshape(n_samples*n_spins, 2)
    
#     A = AB_list[:,0].reshape(n_samples*n_spins, 1)
#     B = AB_list[:,1].reshape(n_samples*n_spins, 1)     # a,1       a = n_samples*n_spins
    
#     w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)  # a,1
#     mz = (A + wL_value) / w_tilda                 # a,1
#     mx = B / w_tilda                              # a,1

#     alpha = w_tilda * time_table.reshape(1, n_time_values)    # a,b    b=len(time_table)
#     beta = wL_value * time_table.reshape(1, n_time_values)    # 1,b

#     phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))  # a,b
#     K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))               # a,b
#     K2 = 1 + np.cos(phi)                                        # a,b
#     K = pow(mx,2) * (K1 / K2)                                   # a,b
#     M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)       # a,b

#     M_list_temp = M_list_temp.reshape(n_samples, n_spins, n_time_values)
#     M_list = np.prod(M_list_temp, axis=1)    #  c, b         c = n_samples

#     return M_list

# def M_list_return_bulk_8_16_32(time_table, wL_value, AB_many_list):
    
#     n_pulses = [8, 16, 32]
#     n_samples = len(AB_many_list)
#     n_spins = len(AB_many_list[0])
#     n_time_values = len(time_table)
#     AB_list = AB_many_list.reshape(n_samples*n_spins, 2)
    
#     A = AB_list[:,0].reshape(n_samples*n_spins, 1)
#     B = AB_list[:,1].reshape(n_samples*n_spins, 1)     # a,1       a = n_samples*n_spins
    
#     w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)  # a,1
#     mz = (A + wL_value) / w_tilda                 # a,1
#     mx = B / w_tilda                              # a,1

#     alpha = w_tilda * time_table.reshape(1, n_time_values)    # a,b    b=len(time_table)
#     beta = wL_value * time_table.reshape(1, n_time_values)    # 1,b

#     phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))  # a,b
#     K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))               # a,b
#     K2 = 1 + np.cos(phi)                                        # a,b
#     K = pow(mx,2) * (K1 / K2)                                   # a,b

#     M_list = np.zeros((n_samples, 3, n_time_values))
#     for idx, n_pulse in enumerate(n_pulses):
#         M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)       # a,b
#         M_list_temp = M_list_temp.reshape(n_samples, n_spins, n_time_values)
#         M_list[:, idx, :] = np.prod(M_list_temp, axis=1)    #  c, b         c = n_samples

#     return M_list\\\\\\\\\\\\\

def M_list_return(time_table, wL_value, AB_list, n_pulse):  # AB_list must be an array, not list

    AB_list = np.array(AB_list)
    A = AB_list[:,0].reshape(len(AB_list), 1)
    B = AB_list[:,1].reshape(len(AB_list), 1)     # a,1     a = len(AB_list)
    
    w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)  # a,1
    mz = (A + wL_value) / w_tilda                 # a,1
    mx = B / w_tilda                              # a,1

    alpha = w_tilda * time_table.reshape(1, len(time_table))    # a,b    b=len(time_table)
    beta = wL_value * time_table.reshape(1, len(time_table))    # 1,b

    phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))  # a,b
    K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))               # a,b
    K2 = 1 + np.cos(phi)                                        # a,b
    K = pow(mx,2) * (K1 / K2)                                   # a,b
    M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)       # a,b
    M_list = np.prod(M_list_temp, axis=0)

    return M_list
    # 이 함수가 time_table을 넣으면 그에 맞는 M값 array를 return.
    # wL value, n_pulse는 외부 자기장이나 실험 조건중에 하나임. 상수로 처음에 설정해줌.

# def M_list_8_16_32_return(time_table, wL_value, AB_list):
    
#     N_pulse = [8, 16, 32]
#     M_list_8_16_32 = np.ones((3, len(time_table)))
#     for [A, B] in AB_list:                            ## 이 부분은 매우 중요. 2pi가 곱해져야 제대로 나옴. 중복계산도 고려해봐야함!!
        
#         w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)
#         mz = (A + wL_value) / w_tilda
#         mx = B / w_tilda

#         alpha = w_tilda * time_table
#         beta = wL_value * time_table

#         phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))
#         K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))
#         K2 = 1 + np.cos(phi)
#         K = pow(mx,2) * (K1 / K2)
#         for idx, n_pulse in enumerate(N_pulse):
#             M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)
#             M_list_8_16_32[idx] *= M_list_temp

#     return M_list_8_16_32

def M_list_8_16_32_return(time_table, wL_value, AB_list):

    N_pulse = [8,16,32]
    M_list = np.zeros((3, len(time_table)))
    A = AB_list[:,0].reshape(len(AB_list), 1)
    B = AB_list[:,1].reshape(len(AB_list), 1)
    
    w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)
    mz = (A + wL_value) / w_tilda
    mx = B / w_tilda

    alpha = w_tilda * time_table.reshape(1, len(time_table))
    beta = wL_value * time_table.reshape(1, len(time_table))

    phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))
    K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))
    K2 = 1 + np.cos(phi)
    K = pow(mx,2) * (K1 / K2)

    for idx, n_pulse in enumerate(N_pulse):
        M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)
        M_list[idx] = np.prod(M_list_temp, axis=0)

    return M_list


def M_list_all_return(time_table, wL_value, AB_list, n_pulse):
    
    M_list = np.ones([len(AB_list)+1, len(time_table)])
    for idx, [A, B] in enumerate(AB_list):
        
        w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)
        mz = (A + wL_value) / w_tilda
        mx = B / w_tilda

        alpha = w_tilda * time_table
        beta = wL_value * time_table

        phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))
        K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))
        K2 = 1 + np.cos(phi)
        K = pow(mx,2) * (K1 / K2)
        M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)
            
        M_list[idx] *= M_list_temp
        M_list[-1] *= M_list_temp
        
    return M_list

    
def Px_list_return(M_list, time_index, time_table, px_mean_value_at_time, Gaussian_co=False):

    if Gaussian_co == False:
        px_list = (M_list + 1) / 2
        return px_list
    else:
        M_list = gaussian_slope(M_list, time_table, time_index, px_mean_value_at_time)
        px_list = (M_list + 1) / 2
        return px_list

def get_target_slope(start_idx, target_px, m_batch=20):
    total_target_M_list_8_16_32_slope = np.zeros(target_px.shape)
    n_samples = target_px.shape[0]
    for m in range(start_idx, start_idx+n_samples, m_batch):
        total_target_M_list_8_16_32_slope[m:m+m_batch,0,:], slope = gaussian_slope_px((2*target_px[m:m+m_batch,0,:])-1, time_table, TIME_INDEX_8, MEAN_VALUE_8)
        total_target_M_list_8_16_32_slope[m:m+m_batch,1,:], slope = gaussian_slope_px((2*target_px[m:m+m_batch,1,:])-1, time_table, TIME_INDEX_16, MEAN_VALUE_16)
        total_target_M_list_8_16_32_slope[m:m+m_batch,2,:], slope = gaussian_slope_px((2*target_px[m:m+m_batch,2,:])-1, time_table, TIME_INDEX_32, MEAN_VALUE_32)
    return total_target_M_list_8_16_32_slope

def get_none_target_slope(start_idx, none_target_px, m_batch=20):
    n_samples = none_target_px.shape[0]
    total_none_target_M_list_8_16_32_slope = np.zeros(none_target_px.shape)
    for m in range(start_idx, start_idx+n_samples, m_batch):
        total_none_target_M_list_8_16_32_slope[m:m+m_batch,0,:], slope = gaussian_slope_px((2*none_target_px[m:m+m_batch,0,:])-1, time_table, TIME_INDEX_8, MEAN_VALUE_8)
        total_none_target_M_list_8_16_32_slope[m:m+m_batch,1,:], slope = gaussian_slope_px((2*none_target_px[m:m+m_batch,1,:])-1, time_table, TIME_INDEX_16, MEAN_VALUE_16)
        total_none_target_M_list_8_16_32_slope[m:m+m_batch,2,:], slope = gaussian_slope_px((2*none_target_px[m:m+m_batch,2,:])-1, time_table, TIME_INDEX_32, MEAN_VALUE_32)
    return total_none_target_M_list_8_16_32_slope

def norm_ydata(exp_data):

    mean = 0.94275
    amp = 0.14131
    y_max = mean + amp
    y_min = mean - amp

    ydata_f = (y_max - exp_data) / (2*amp)
    if exp_data[0] > mean:
        ydata_f =1-ydata_f
    else:
        ydata_f = ydata_f
    return ydata_f

def noise_generator_8_16_32(px_table, time_table):
    choice = np.random.randint(0,2)
    uni_min = 0.02
    uni_max = 0.13
    normal_min = 0.02
    normal_max = 0.065

    if choice == 0:
        noise_width = 2
        max_noise = np.random.uniform(uni_min, uni_max)
        uniform_noise = np.random.uniform(-max_noise, max_noise, size=len(time_table))
        noise_list = np.zeros(len(time_table))
        even_indices = range(0, len(time_table), noise_width)
        odd_indices = range(1, len(time_table)-2, noise_width)
        noise_list[even_indices] = uniform_noise[even_indices]
        mid_point = (uniform_noise[even_indices][:-1] + uniform_noise[even_indices][1:]) / 2
        noise_list[odd_indices] = mid_point

    elif choice == 1:
        noise_width = 2
        std = np.random.uniform(normal_min, normal_max)
        noise_max_tolerance = 0.16
        uniform_noise = np.random.normal(0., std, size=len(time_table))
        noise_list = np.zeros(len(time_table))
        even_indices = range(0, len(time_table), noise_width)
        odd_indices = range(1, len(time_table)-2, noise_width)
        noise_list[even_indices] = uniform_noise[even_indices]
        mid_point = (uniform_noise[even_indices][:-1] + uniform_noise[even_indices][1:]) / 2
        noise_list[odd_indices] = mid_point
        noise_list[noise_list > noise_max_tolerance] = noise_max_tolerance
        noise_list[noise_list < -noise_max_tolerance] = -noise_max_tolerance
    
    noise_list = noise_list.reshape((1, -1))
    px_table_noise = px_table + noise_list
    return px_table_noise

def noise_generator(px_table, time_table):
    choice = np.random.randint(0,2)
    uni_min = 0.02
    uni_max = 0.13
    normal_min = 0.02
    normal_max = 0.065

    if choice == 0:
        noise_width = 2
        max_noise = np.random.uniform(uni_min, uni_max)
        uniform_noise = np.random.uniform(-max_noise, max_noise, size=len(time_table))
        noise_list = np.zeros(len(time_table))
        even_indices = range(0, len(time_table), noise_width)
        odd_indices = range(1, len(time_table)-2, noise_width)
        noise_list[even_indices] = uniform_noise[even_indices]
        mid_point = (uniform_noise[even_indices][:-1] + uniform_noise[even_indices][1:]) / 2
        noise_list[odd_indices] = mid_point

    elif choice == 1:
        noise_width = 2
        std = np.random.uniform(normal_min, normal_max)
        noise_max_tolerance = 0.16
        uniform_noise = np.random.normal(0., std, size=len(time_table))
        noise_list = np.zeros(len(time_table))
        even_indices = range(0, len(time_table), noise_width)
        odd_indices = range(1, len(time_table)-2, noise_width)
        noise_list[even_indices] = uniform_noise[even_indices]
        mid_point = (uniform_noise[even_indices][:-1] + uniform_noise[even_indices][1:]) / 2
        noise_list[odd_indices] = mid_point
        noise_list[noise_list > noise_max_tolerance] = noise_max_tolerance
        noise_list[noise_list < -noise_max_tolerance] = -noise_max_tolerance
    
    px_table_noise = px_table + noise_list
    return px_table_noise


def convert_to_4000(px_table, time_table):

    front = px_table[:-1]
    back = px_table[1:]
    mid_1 = (front + back) / 2
    mid_2 = (front + mid_1) / 2
    mid_3 = (mid_1 + back) / 2
    converted_list = np.zeros((len(px_table)-1)*4)
    indices = np.arange(0, len(converted_list), 4)

    converted_list[indices] = px_table[:-1]
    converted_list[indices+1] = mid_2
    converted_list[indices+2] = mid_1
    converted_list[indices+3] = mid_3
    converted_list[-1] = converted_list[-2] + np.random.uniform(-0.03, 0.03)

    return converted_list

def save_figures_comparison(time_table, norm_exp_data16, px_list_slope_16, norm_exp_data32, px_list_slope_32, test_array, idx, index):
    fig = plt.figure(figsize=(7, 5), facecolor='w')        
    ax = fig.add_subplot(211)
    ax.set_ylim(0.0, 1.1)
    ax.plot(time_table, norm_exp_data16, 'o-', label='N_pulse_16', markersize=1., lw=0.7)
    ax.plot(time_table, px_list_slope_16, label=str(idx)+str(test_array[index]), lw=0.8)
    ax.legend(loc='lower left', fontsize=6)

    ax2 = fig.add_subplot(212)
    ax2.set_ylim(0.0, 1.1)
    ax2.plot(time_table[:879], norm_exp_data32[:879], 'o-', label='N_pulse_32', markersize=1., lw=0.7)
    ax2.plot(time_table[:879], px_list_slope_32[:879], label=str(idx)+str(test_array[index]), lw=0.8)
    ax2.legend(loc='lower left', fontsize=6)

    plt.savefig('.././images_to_movie_A/{}_{}.jpg'.format(idx, index), format='jpg', dpi=200)
    plt.close('all')
    fig.clear()
    fig.clf()
    gc.collect()

def remove_same_ABvalue(test_list):
    test_dict = {}
    for idx, i in enumerate(test_list):
        test_dict[idx] = i

    indices_dict = {}
    for i in test_dict:
        j_temp = []
        for j in test_dict:
            if i != j:
                if test_dict[i][0] == test_dict[j][0]:
                    j_temp.append(j)
        indices_dict[i] = j_temp

    nosame_value_list = []
    for idx in indices_dict:
        if len(indices_dict[idx]) == 0:
            nosame_value_list.append(test_list[idx])
        else:
            if idx < indices_dict[idx][0]:
                nosame_value_list.append(test_list[idx])

    return nosame_value_list

def sort_AB_list(input_list):

    indices = np.argsort(input_list[:,0])            
    sorted_list = input_list[indices,:]
    
    return sorted_list

def plot_pred(n_samples, y_pred, y_eval_data=None):

    fig = plt.figure(figsize=(40,30), facecolor='w')
    ax = fig.add_subplot(311)
    ax.set_xlabel('AB_pair', color='k', fontsize=20)
    ax.set_ylabel('count', color='k', fontsize=20)
    ax.tick_params(axis='x', colors='k', labelsize=15)
    ax.tick_params(axis='y', colors='k', labelsize=15)

    idx = np.random.randint(0, 1023)
    ax.plot(y_pred[idx], 'o-', color='#DC5320', markersize=10, label='pred')
    ax.plot(y_eval_data[idx], color='#3069A0', label='eval')

    ax = fig.add_subplot(312)
    idx = np.random.randint(0, 1023)
    ax.set_xlabel('AB_pair', color='k', fontsize=20)
    ax.set_ylabel('count', color='k', fontsize=20)
    ax.tick_params(axis='x', colors='k', labelsize=15)
    ax.tick_params(axis='y', colors='k', labelsize=15)

    ax.plot(y_pred[idx], 'o-', color='#DC5320', markersize=10, label='pred')
    ax.plot(y_eval_data[idx], color='#3069A0', label='eval')

    ax = fig.add_subplot(313)
    idx = np.random.randint(0, 1023)
    ax.set_xlabel('AB_pair', color='k', fontsize=20)
    ax.set_ylabel('count', color='k', fontsize=20)
    ax.tick_params(axis='x', colors='k', labelsize=15)
    ax.tick_params(axis='y', colors='k', labelsize=15)

    ax.plot(y_pred[idx], 'o-', color='#DC5320', markersize=10, label='pred')
    ax.plot(y_eval_data[idx], color='#3069A0', label='eval')

def plot_exp_pred(y_exp_pred):

    fig = plt.figure(figsize=(20,10), facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_xlabel('AB_pair', color='k', fontsize=20)
    ax.set_ylabel('count', color='k', fontsize=20)
    ax.tick_params(axis='x', colors='k', labelsize=15)
    ax.tick_params(axis='y', colors='k', labelsize=15)

    ax.plot(y_exp_pred[0], 'o-', color='#DC5320', markersize=10)


def noise_generator_4000(px_table, time_table):
    choice = np.random.randint(0,2)

    if choice == 0:
        noise_width = 8
        max_noise = np.random.uniform(0.02, 0.13)
        uniform_noise = np.random.uniform(-max_noise, max_noise, size=len(time_table)+1)
        noise_list = np.zeros(len(time_table)+1)

        width_indices = np.arange(0, len(time_table)+1, noise_width)
        noise_list[width_indices] = uniform_noise[width_indices]
        front = uniform_noise[width_indices[:-1]]
        back = uniform_noise[width_indices[1:]]

        noise_list[width_indices[:-1] + 1] = (front + back) * (1 / 8)
        noise_list[width_indices[:-1] + 2] = (front + back) * (2 / 8)
        noise_list[width_indices[:-1] + 3] = (front + back) * (3 / 8)
        noise_list[width_indices[:-1] + 4] = (front + back) * (4 / 8)
        noise_list[width_indices[:-1] + 5] = (front + back) * (5 / 8)
        noise_list[width_indices[:-1] + 6] = (front + back) * (6 / 8)
        noise_list[width_indices[:-1] + 7] = (front + back) * (7 / 8)

    elif choice == 1:

        noise_width = 8
        std = np.random.uniform(0.02, 0.065)
        noise_max_tolerance = 0.16
        uniform_noise = np.random.normal(0., std, size=len(time_table)+1)
        noise_list = np.zeros(len(time_table)+1)

        width_indices = np.arange(0, len(time_table)+1, noise_width)
        noise_list[width_indices] = uniform_noise[width_indices]
        front = uniform_noise[width_indices[:-1]]
        back = uniform_noise[width_indices[1:]]

        noise_list[width_indices[:-1] + 1] = (front + back) * (1 / 8)
        noise_list[width_indices[:-1] + 2] = (front + back) * (2 / 8)
        noise_list[width_indices[:-1] + 3] = (front + back) * (3 / 8)
        noise_list[width_indices[:-1] + 4] = (front + back) * (4 / 8)
        noise_list[width_indices[:-1] + 5] = (front + back) * (5 / 8)
        noise_list[width_indices[:-1] + 6] = (front + back) * (6 / 8)
        noise_list[width_indices[:-1] + 7] = (front + back) * (7 / 8)

        noise_list[noise_list > noise_max_tolerance] = noise_max_tolerance
        noise_list[noise_list < -noise_max_tolerance] = -noise_max_tolerance

    px_table_noise = px_table + noise_list[:-1]
    return px_table_noise

def getsizeof_variable(vriable, suffix='B'):
    vriable_size = sys.getsizeof(vriable)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(vriable_size) < 1024.0:
            return "%3.1f%s%s" % (vriable_size, unit, suffix)
        vriable_size /= 1024.0
    return "%.1f%s%s" % (vriable_size, 'Yi', suffix)

# for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
#                          key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

def plot_history(history, save_directory, start_idx=0):
    if type(history) == list:
        total_loss = []
        total_val_loss = []

        for temp_hist in history:
            total_loss += temp_hist.history['loss']
            total_val_loss += temp_hist.history['val_loss']

    plt.figure(facecolor='w', figsize=(20,10))
    plt.plot(total_loss[start_idx:], label='loss')
    plt.plot(total_val_loss[start_idx:], label='val_loss')
    plt.legend()

    current_time = time.localtime()[:]
    time_name = str(current_time[0])+'_'+str(current_time[1])+'_'+str(current_time[2])+'_'+str(current_time[3])+'_'+str(current_time[4])+'_'
    save_name = save_directory + time_name + 'history.npy'
    print("History Saved!")
    np.save(save_name, [total_loss, total_val_loss])
    # history를 plot해주는 함수

def total_training_time(time_list):
    total_time_list = []
    for line in time_list:
        total_time_list += line
    total_time_list = np.array(total_time_list)
    total_time = np.sum(total_time_list)
    return total_time
    # total training time을 return

def change_model_lr(model, lr):
    current_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, current_lr*lr)
    print('current_lr: %.4f, changed_lr: %.4f' % (current_lr, K.get_value(model.optimizer.lr)))
    # learning rate을 중간에 바꿔주는 함수

def get_error(px_list, exp_data, threshold=0.06):
    px_mean = np.mean(px_list)
    indices = px_list<(px_mean-threshold)
    filtered_px_list = px_list[indices]
    filtered_exp_data = exp_data[indices]
    SquaredError = np.sum(np.square(filtered_px_list-filtered_exp_data))
    error = np.sqrt(SquaredError)
    return error
    # model에서 prediction한 후, px_list(CPMG simulation graph) 와 실험치의 RMS를 return
    # 나중에는 이 값을 다 취합하여 최소 RMS를 구한다.

def delete_same_value_with_avg(list_A, list_B): 
    temp_A = list_A.reshape(-1, 1)
    temp_B = list_B.reshape(-1, 1)
    total_temp = np.concatenate((temp_A, temp_B), axis=1)

    df = pd.DataFrame(total_temp)
    df2 = df.groupby(0).mean().reset_index()

    return df2

def get_filtered_idx(M_value, margin=10, width_threshold=100): # filter threshold : (mean-std)
    mean = np.mean(M_value)
    std = np.std(M_value)
    indices = np.where(M_value < (mean-std))[0]
    margin = margin                        # left-right margins : 10 points
    width_threshold = width_threshold      # distance between peaks
    
    temp_idx = []
    temp_idx.append(indices[0]-margin)  
    for i, idx in enumerate(indices):
        if i != (len(indices)-1):
            temp1 = indices[i+1] - idx
        if temp1 > width_threshold:          
            temp_idx.append(indices[i]+margin)
            if i > (len(indices)-2):break
            temp_idx.append(indices[i+1]-margin)
    temp_idx.append(indices[-1]+margin)
    
    filtered_idx = []
    for i in range(0, len(temp_idx)-1, 2):
        if (i == (len(M_value)-1)) or (i == len(M_value)): break
        filtered_idx += [i for i in range(temp_idx[i], temp_idx[i+1]+1)]

    filtered_arr = np.array(filtered_idx)
    filtered_arr = filtered_arr[(filtered_arr>-1) & (filtered_arr<len(M_value))]

    return filtered_arr

def delete_same_value_with_avg(list_A, list_B): 
    temp_A = list_A.reshape(-1, 1)
    temp_B = list_B.reshape(-1, 1)
    total_temp = np.concatenate((temp_A, temp_B), axis=1)

    df = pd.DataFrame(total_temp)
    df2 = df.groupby(0).mean().reset_index()

    return df2

def history_plotly(hist):
    init_notebook_mode(connected=True)

    loss = []
    val_loss = []
    acc = []
    val_acc = []

    for history_temp in hist:
        loss += history_temp['loss']
        val_loss += history_temp['val_loss']
        acc += history_temp['acc']
        val_acc += history_temp['val_acc']

    iplot([go.Line(y=loss, name='loss'),
           go.Line(y=val_loss, name='val_loss'),
           go.Line(y=acc, name='acc'),
           go.Line(y=val_acc, name='val_acc')], show_link=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gen_AB_candidates_arrays(*args):
    total_ab_array, A_candidate_max, A_candidate_min, B_candidate_max, B_candidate_min, A_small_max, A_small_min, B_small_max, B_small_min, A_max, A_min, B_max, B_min, A_step, B_step, index = args
    
    if index % 3 == 0:
        indices = np.where((total_ab_array[:,0]>=A_min) & (total_ab_array[:,0]<=A_max) & (total_ab_array[:,1]>=B_min) & (total_ab_array[:,1]<=B_max))
        target_array = total_ab_array[indices]

        candidate_boolen_index = ((total_ab_array[:,1]>=B_candidate_min) & (total_ab_array[:,1]<=B_candidate_max))   \
                                & ((total_ab_array[:,0]>=A_candidate_min) & (total_ab_array[:,0]<=A_candidate_max))  \
                                & (((total_ab_array[:,1]<=((B_min-2000)-B_step)) | (total_ab_array[:,1]>=((B_max+2000)+B_step)))   \
                                | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        AB_candidate_array = total_ab_array[candidate_boolen_index]

        small_AB_boolen_index = ((total_ab_array[:,1]>=B_small_min) & (total_ab_array[:,1]<=B_small_max))   \
                                 & ((total_ab_array[:,0]>=A_small_min) & (total_ab_array[:,0]<=A_small_max)) \
                                 & (((total_ab_array[:,1]<=((B_min-2000)-B_step)) | (total_ab_array[:,1]>=((B_max+2000)+B_step)))      \
                                 | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        small_AB_array = total_ab_array[small_AB_boolen_index]
        print("Mid_generated", end=' | ')

    elif index % 3 == 1:
        indices = np.where((total_ab_array[:,0]>=A_min) & (total_ab_array[:,0]<=A_max) & (total_ab_array[:,1]>=B_min) & (total_ab_array[:,1]<=B_max))
        target_array = total_ab_array[indices]

        candidate_boolen_index = ((total_ab_array[:,1]>=B_candidate_min) & (total_ab_array[:,1]<=B_candidate_max))   \
                                & ((total_ab_array[:,0]>=A_candidate_min) & (total_ab_array[:,0]<=A_candidate_max))  \
                                & (((total_ab_array[:,1]<=((B_min-4000)-B_step)) | (total_ab_array[:,1]>=(B_max+B_step)))  \
                                | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        AB_candidate_array = total_ab_array[candidate_boolen_index]        

        small_AB_boolen_index = ((total_ab_array[:,1]>=B_small_min) & (total_ab_array[:,1]<=B_small_max))   \
                                 & ((total_ab_array[:,0]>=A_small_min) & (total_ab_array[:,0]<=A_small_max)) \
                                 & (((total_ab_array[:,1]<=((B_min-4000)-B_step)) | (total_ab_array[:,1]>=(B_max+B_step)))      \
                                 | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        small_AB_array = total_ab_array[small_AB_boolen_index]
        print("Bottom_generated", end=' | ')

    else:
        indices = np.where((total_ab_array[:,0]>=A_min) & (total_ab_array[:,0]<=A_max) & (total_ab_array[:,1]>=B_min) & (total_ab_array[:,1]<=B_max))
        target_array = total_ab_array[indices]

        candidate_boolen_index = ((total_ab_array[:,1]>=B_candidate_min) & (total_ab_array[:,1]<=B_candidate_max))   \
                                & ((total_ab_array[:,0]>=A_candidate_min) & (total_ab_array[:,0]<=A_candidate_max))  \
                                & (((total_ab_array[:,1]<=(B_min-B_step)) | (total_ab_array[:,1]>=((B_max+4000)+B_step)))  \
                                | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        AB_candidate_array = total_ab_array[candidate_boolen_index]
        
        small_AB_boolen_index = ((total_ab_array[:,1]>=B_small_min) & (total_ab_array[:,1]<=B_small_max))   \
                                 & ((total_ab_array[:,0]>=A_small_min) & (total_ab_array[:,0]<=A_small_max)) \
                                 & (((total_ab_array[:,1]<=(B_min-B_step)) | (total_ab_array[:,1]>=((B_max+4000)+B_step)))      \
                                 | ((total_ab_array[:,0]<=(A_min-A_step)) | (total_ab_array[:,0]>=(A_max+A_step))))
        small_AB_array = total_ab_array[small_AB_boolen_index]
        print("Top_generated", end=' | ')
        
    # 이거 반드시 확인! 매우 중요. 2*np.pi 곱하지 않으면, 당연히 Training이 안됨.
    AB_candidate_array *= 2 * np.pi
    target_array *= 2 * np.pi
    small_AB_array *= 2 * np.pi

    return target_array, AB_candidate_array, small_AB_array

def gen_target_nonetarget_ABlists(target_array, AB_candidate_array, small_AB_array, n_of_spins, n_of_small_spins, n_of_samples_per_target):
    target_temp = np.zeros((n_of_spins-n_of_small_spins, 2))
    target_AB_included = np.zeros((len(target_array), n_of_samples_per_target, n_of_spins, 2))
    none_target_AB = np.zeros((len(target_array), n_of_samples_per_target, n_of_spins, 2))

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
    return target_AB_included, none_target_AB

    
def gen_ABlists(target_array, AB_candidate_array, small_AB_array, n_of_spins, n_of_small_spins, n_of_samples_per_target):
    total_ABlists = np.zeros((10, (n_of_samples_per_target//4)*16, n_of_spins, 2))

    for i in range(len(target_array)):
        target_temp = np.zeros(((n_of_samples_per_target//4)*16, n_of_spins, 2))
        target_array_temp = target_array[i]
        
        for idx, target_AB in enumerate(target_array_temp):
            target_temp[idx*n_of_samples_per_target:(idx+1)*n_of_samples_per_target,0,:] = target_AB
            for j in range(n_of_samples_per_target):
                indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-1)
                while len(set(indices_candi)) != n_of_spins-n_of_small_spins-1:
                    indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-1)
                target_temp[idx*n_of_samples_per_target + j, 1:n_of_spins-n_of_small_spins] = AB_candidate_array[indices_candi]

                indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
                while len(set(indices_candi)) != n_of_small_spins:
                    indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
                small_AB_temp = small_AB_array[indices_candi]
                target_temp[idx*n_of_samples_per_target + j, n_of_spins-n_of_small_spins:] = small_AB_temp
        np.random.shuffle(target_temp)
        total_ABlists[i] = target_temp

    index_list = [[0, 0],[0, 1],[0, 2],[1, 1],[1, 2],[2, 2]]
    for count, target_index in enumerate(index_list):
        for idx1, AB1 in enumerate(target_array[target_index[0]]):
            target_temp[idx1*n_of_samples_per_target:(idx1+1)*(n_of_samples_per_target),0,:] = AB1
            for idx2, AB2 in enumerate(target_array[target_index[1]]):
                target_temp[idx1*n_of_samples_per_target + idx2*(n_of_samples_per_target//4):idx1*n_of_samples_per_target + (idx2+1)*(n_of_samples_per_target//4),1,:] = AB2
                for j in range(n_of_samples_per_target//4):
                    indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-2)
                    while len(set(indices_candi)) != n_of_spins-n_of_small_spins-2:
                        indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins-2)
                    target_temp[idx*n_of_samples_per_target + j, 2:n_of_spins-n_of_small_spins] = AB_candidate_array[indices_candi]

                    indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
                    while len(set(indices_candi)) != n_of_small_spins:
                        indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
                    small_AB_temp = small_AB_array[indices_candi]
                    target_temp[idx*n_of_samples_per_target + j, n_of_spins-n_of_small_spins:] = small_AB_temp
        np.random.shuffle(target_temp)
        total_ABlists[count+3] = target_temp
        print(target_temp.shape, target_index)        

    for j in range(4*n_of_samples_per_target):
        indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins)
        while len(set(indices_candi)) != n_of_spins-n_of_small_spins:
            indices_candi = np.random.randint(len(AB_candidate_array), size=n_of_spins-n_of_small_spins)
        non_target_temp = AB_candidate_array[indices_candi]

        indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        while len(set(indices_candi)) != n_of_small_spins:
            indices_candi = np.random.randint(len(small_AB_array), size=n_of_small_spins)
        small_AB_temp = small_AB_array[indices_candi]
        target_temp[j, :n_of_spins-n_of_small_spins] = non_target_temp
        target_temp[j, n_of_spins-n_of_small_spins:] = small_AB_temp
    total_ABlists[-1] = target_temp

    return total_ABlists

def gen_X_train(x_batch_ABdata, time_data, WL_VALUE, N_PULSE_32):
    x_train_temp = np.zeros((x_batch_ABdata.shape[0], time_data.shape[0]))
    for idx, ABs in enumerate(x_batch_ABdata):
        x_train_temp[idx] = M_list_return(time_data, WL_VALUE, ABs, N_PULSE_32)
    return x_train_temp
