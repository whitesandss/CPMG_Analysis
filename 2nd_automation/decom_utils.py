import pandas as pd
import numpy as np
import os, sys
import gc
import time
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# 원래는 spin함수의 확률 신호는 시간에 따라 exponetial 하게 decoherence함.
# 그 감소하는 경향성을 return하는 것.
def gaussian_slope(time_data, time_index, px_mean_value_at_time, M_list=None):
    m_value_at_time = (px_mean_value_at_time * 2) - 1
    Gaussian_co = -time_data[time_index] / np.log(m_value_at_time)

    slope = np.exp(-time_data / Gaussian_co)
    if M_list != None:
        M_list_slope = M_list * slope
        px_list_slope = (1 + M_list_slope) / 2
        return px_list_slope, slope
    return slope

# time을 넣으면 M값 return.
# wL value, n_pulse는 외부 자기장이나 실험 조건.
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
    
# delete same values of an array (inplace = average value)
def delete_same_value_with_avg(list_A, list_B): 
    temp_A = list_A.reshape(-1, 1)
    temp_B = list_B.reshape(-1, 1)
    total_temp = np.concatenate((temp_A, temp_B), axis=1)

    df = pd.DataFrame(total_temp)
    df2 = df.groupby(0).mean().reset_index()

    return df2

# Normalize the experimental data
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

# model에서 prediction한 후, px_list(CPMG simulation graph) 와 실험치의 RMS를 return
# 나중에는 이 값을 다 취합하여 최소 RMS를 구한다.
def get_error(px_list, exp_data, threshold=False):
    px_mean = np.mean(px_list)
    px_std = np.std(px_list)
    
    if threshold>0:
        indices = px_list<(px_mean-threshold)
    else:
        indices = px_list<(px_mean-px_std)
        
    filtered_px_list = px_list[indices]
    filtered_exp_data = exp_data[indices]
    SquaredError = np.sum(np.square(filtered_px_list-filtered_exp_data))
    error = np.sqrt(SquaredError)
    return error

def get_filtered_idx(M_value, margin=10, width_threshold=100, cut_threshold=False): # filter threshold : (mean-std)
    mean = np.mean(M_value)
    std = np.std(M_value)
    if cut_threshold == False:
        indices = np.where(M_value < (mean-std))[0] # cut_threshold = mean-std
    else:
        indices = np.where(M_value < cut_threshold)[0]

    if margin == 0:
        filtered_arr = np.array(indices)
        
    else:
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


# Generate Data
def gen_X_train(x_batch_ABdata, time_data, WL_VALUE, N_PULSE_32):
    x_train_temp = np.zeros((x_batch_ABdata.shape[0], time_data.shape[0]))
    for idx, ABs in enumerate(x_batch_ABdata):
        x_train_temp[idx] = M_list_return(time_data, WL_VALUE, ABs, N_PULSE_32)
    return x_train_temp

# return fitting AB_valuse w.r.t fitting parameters-(A_range, B_range, A_step, B_step), units of AB spin values : Hz
def get_fitting_values(AB_value, A_range, B_range, A_step, B_step):
    A, B = AB_value
    A_lists = np.arange(A-A_range, A+A_range+0.1, A_step)
    B_lists = np.arange(B-B_range, B+B_range+0.1, B_step)
    AB_lists = np.zeros((len(A_lists)*len(B_lists), 2))

    count = 0
    for idx, A_value in enumerate(A_lists):
        A_temp = np.full((len(B_lists), 1), A_value)
        AB_lists[idx*len(B_lists):(idx+1)*len(B_lists)] = np.concatenate((A_temp, B_lists.reshape(-1, 1)), axis=1)
    AB_lists *= 2*np.pi
    
    total_mse_error = np.zeros((AB_lists.shape[0], AB_lists.shape[1]+1))
    for idx, AB_test in enumerate(AB_lists):
        px_value = (1 + M_list_return(time_data, WL_VALUE, [AB_test], N_PULSE_32)) / 2
        mse_error = get_error(px_value, exp_data, threshold=False)
        total_mse_error[idx] = np.round(mse_error, 5), AB_test[0]/2/np.pi, AB_test[1]/2/np.pi
    
    return total_mse_error