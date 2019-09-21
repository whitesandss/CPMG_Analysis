# A: 35~37, B: 18~36

import numpy as np
import time
import matplotlib.pyplot as plt
import os, glob
import gc
import sys
sys.path.insert(0, '/home/sonic/Coding/Git/CPMG_Analysis/2nd_automation/')
from decom_utils import *
import torch
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary
from multiprocessing import Pool
from sklearn.utils import shuffle

TIME_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]

MAGNETIC_FIELD = 403.7139663551402            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000               # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32

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

# 원하는 부분만 AB_list를 만들고 싶을때

heatmap_n_rows = len(B_range_list)
target_indices = []
A_index_min = 95  # 원하는 A영역
A_index_max = 97
B_index_min = 8   # 원하는 B영역
B_index_max = 16

for i in range(len(AB_dic)):
    if (i//heatmap_n_rows >= A_index_min) & (i//heatmap_n_rows <= A_index_max):
        if (i%heatmap_n_rows >= B_index_min) & (i%heatmap_n_rows <= B_index_max):
            target_indices.append(i)

# 뽑을 spin 개수 및 target 스핀 한 개당 만들 data의 개수
n_of_spins = 39
n_of_small_spins = 20 
n_of_samples_per_target = 40000

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

AB_range_list_arr = np.array([[[[35000, 36000], [18000, 20000]], [[35000, 36000], [20000, 22000]], [[35000, 36000], [22000, 24000]]],
                              [[[35000, 36000], [24000, 26000]], [[35000, 36000], [26000, 28000]], [[35000, 36000], [28000, 30000]]],
                              [[[35000, 36000], [30000, 32000]], [[35000, 36000], [32000, 34000]], [[35000, 36000], [34000, 36000]]],
                              [[[36000, 37000], [18000, 20000]], [[36000, 37000], [20000, 22000]], [[36000, 37000], [22000, 24000]]],
                              [[[36000, 37000], [24000, 26000]], [[36000, 37000], [26000, 28000]], [[36000, 37000], [28000, 30000]]],
                              [[[36000, 37000], [30000, 32000]], [[36000, 37000], [32000, 34000]], [[36000, 37000], [34000, 36000]]],
                              [[[37000, 38000], [18000, 20000]], [[37000, 38000], [20000, 22000]], [[37000, 38000], [22000, 24000]]],
                              [[[37000, 38000], [24000, 26000]], [[37000, 38000], [26000, 28000]], [[37000, 38000], [28000, 30000]]],
                              [[[37000, 38000], [30000, 32000]], [[37000, 38000], [32000, 34000]], [[37000, 38000], [34000, 36000]]]])

file_list = glob.glob('/home/sonic/Coding/Git/temp/total_*.npy')
file_list.sort()

POOL_PROCESS = 15
data_batch = 20480
valid_data_batch = 1000
batch_size = 4096
epochs = 500
pool = Pool(processes=POOL_PROCESS)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for idx, AB_range_list in enumerate(AB_range_list_arr):
    
    total_fil_idx = []
    margin=0
    
    for AB in AB_range_list: 
        A, B = AB[0][0] + 500, AB[1][0] + 1000 
        M_value = M_list_return(time_data, WL_VALUE, [[A*2*np.pi,B*2*np.pi]], N_PULSE_32)
        filtered_idx = get_filtered_idx(M_value, margin=margin, width_threshold=70)
        total_fil_idx.append(filtered_idx)
    filtered_indices = list(set(list(total_fil_idx[0])+list(total_fil_idx[1])+list(total_fil_idx[2])))
    
    X_data = np.load(file_list[idx])
    for idx1 in range(X_data.shape[0]):
        globals()["pool_{}".format(idx1)] = pool.apply_async(gen_X_train, [X_data[idx1][:data_batch], time_data[filtered_indices], WL_VALUE, N_PULSE_32])    

    print("Starting to generate data..")
    tic = time.time()

    X_train_arr = np.zeros((X_data.shape[0], data_batch, time_data[filtered_indices].shape[0]))
    X_train_arr[0] = globals()["pool_{}".format(0)].get(timeout=None)
    X_train_arr[1] = globals()["pool_{}".format(1)].get(timeout=None)
    X_train_arr[2] = globals()["pool_{}".format(2)].get(timeout=None)
    X_train_arr[3] = globals()["pool_{}".format(3)].get(timeout=None)
    X_train_arr[4] = globals()["pool_{}".format(4)].get(timeout=None)
    X_train_arr[5] = globals()["pool_{}".format(5)].get(timeout=None)
    X_train_arr[6] = globals()["pool_{}".format(6)].get(timeout=None)
    X_train_arr[7] = globals()["pool_{}".format(7)].get(timeout=None)
    X_train_arr[8] = globals()["pool_{}".format(8)].get(timeout=None)
    X_train_arr[9] = globals()["pool_{}".format(9)].get(timeout=None)
    
    for idx1 in range(X_data.shape[0]):
        globals()["pool_{}".format(idx1)] = pool.apply_async(gen_X_train, [X_data[idx1][data_batch:data_batch+valid_data_batch], time_data[filtered_indices], WL_VALUE, N_PULSE_32])    
        
    X_valid_arr = np.zeros((X_data.shape[0], valid_data_batch, time_data[filtered_indices].shape[0]))
    X_valid_arr[0] = globals()["pool_{}".format(0)].get(timeout=None)
    X_valid_arr[1] = globals()["pool_{}".format(1)].get(timeout=None)
    X_valid_arr[2] = globals()["pool_{}".format(2)].get(timeout=None)
    X_valid_arr[3] = globals()["pool_{}".format(3)].get(timeout=None)
    X_valid_arr[4] = globals()["pool_{}".format(4)].get(timeout=None)
    X_valid_arr[5] = globals()["pool_{}".format(5)].get(timeout=None)
    X_valid_arr[6] = globals()["pool_{}".format(6)].get(timeout=None)
    X_valid_arr[7] = globals()["pool_{}".format(7)].get(timeout=None)
    X_valid_arr[8] = globals()["pool_{}".format(8)].get(timeout=None)
    X_valid_arr[9] = globals()["pool_{}".format(9)].get(timeout=None)
    toc = time.time()
    print("Time consumed: {} (s)".format(toc - tic))
    
    Y_train_arr = np.zeros((X_train_arr.shape[0], X_train_arr.shape[1], 10))
    for idx3 in range(len(Y_train_arr)):
        Y_train_arr[idx3, :, idx3] = 1
    Y_valid_arr = np.zeros((X_valid_arr.shape[0], X_valid_arr.shape[1], 10))
    for idx3 in range(len(Y_valid_arr)):
        Y_valid_arr[idx3, :, idx3] = 1
        
    X_train_arr = X_train_arr.reshape(-1, len(filtered_indices))
    Y_train_arr = Y_train_arr.reshape(-1, Y_train_arr.shape[-1])
    X_valid_arr = X_valid_arr.reshape(-1, len(filtered_indices))
    Y_valid_arr = Y_valid_arr.reshape(-1, Y_valid_arr.shape[-1])

    X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)
    X_valid_arr, Y_valid_arr = shuffle(X_valid_arr, Y_valid_arr)
        
    model = torch.nn.Sequential(
                    nn.Linear(len(filtered_indices), 4096, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.4),
                    nn.Linear(4096, 2048, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(2048, 1024, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(1024, 512, bias=True),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(512, 10, bias=True),
                    nn.Sigmoid()
            ).to(device)
    
    summary(model, (1, len(filtered_indices)))
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    total_batch = len(X_train_arr)

    tic =time.time()
    print("Training Start..")
    model.train()
    total_val_loss = []
    total_acc = []

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(total_batch // batch_size):

            indices = torch.randperm(total_batch)[:batch_size]
            x_train_temp = torch.tensor(X_train_arr[indices], dtype=torch.float).to(device)
            y_train_temp = torch.tensor(Y_train_arr[indices], dtype=torch.float).to(device)

            optimizer.zero_grad()
            hypothesis = model(x_train_temp)
            cost = criterion(hypothesis, y_train_temp)
            cost.backward()
            optimizer.step()

            avg_cost += cost

        print('Epoch:', '%04d' % (epoch + 1), ', Loss =', '{:.5f}'.format(avg_cost / (total_batch // batch_size)), end=', ')

        with torch.no_grad():
            model.eval()
            X_valid = torch.tensor(X_valid_arr, dtype=torch.float, device=device)
            Y_valid = torch.tensor(Y_valid_arr, dtype=torch.float, device=device)

            prediction = model(X_valid)
            val_loss = criterion(prediction, Y_valid)

            bool_pred = torch.argmax(prediction, dim=1, keepdim=True)
            bool_y = torch.argmax(Y_valid, dim=1, keepdim=True)
            # accuracy = torch.tensor(torch.sum(bool_pred == bool_y), dtype=torch.float) / len(X_valid_arr) * 100
            accuracy = torch.sum(bool_pred == bool_y).float() / len(X_valid_arr) * 100
            print('Val_loss: {:.5f}, Accuracy: {:.2f}%'.format(val_loss.item(), accuracy.item()))
            total_val_loss.append(val_loss.cpu().detach().item())
            total_acc.append(accuracy.cpu().detach().item())

        if total_acc[-1] > 90:
            break

    toc =time.time()
    print(toc-tic)
    
    print("Prediction..")
    EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
    exp_data = EXP_DATA[:12000]
    exp_data = 2*exp_data - 1
    exp_tensor = torch.FloatTensor(exp_data[filtered_indices].reshape(1, -1)).to(device)
    exp_tensor.shape
    pred = model(exp_tensor)
    
    np.save("/home/sonic/Coding/Git/temp/save_model/{}_pred_values.npy".format(idx), pred.cpu().detach().numpy())
    np.save("/home/sonic/Coding/Git/temp/save_model/{}_val_loss_{}(s).npy".format(idx, np.round(toc-tic)), np.array(total_val_loss))
    np.save("/home/sonic/Coding/Git/temp/save_model/{}_val_acc.npy".format(idx), np.array(total_acc))
    torch.save(model, '/home/sonic/Coding/Git/temp/save_model/{}_torch_model'.format(idx))
    print("Model Saved.")

    del X_train_arr, X_valid_arr, model, X_valid, Y_valid
    torch.cuda.empty_cache()
    