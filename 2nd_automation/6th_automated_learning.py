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
from torchsummary import summary
from multiprocessing import Pool
from sklearn.utils import shuffle
import torchvision
import torchvision.datasets as dsets

TIME_DATA = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/20190610_time_no_duplicated.npy')*1e-6 
EXP_DATA = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/20190610_exp_data_24200.npy').squeeze() 
exp_data = EXP_DATA[:12000]
time_data = TIME_DATA[:12000] 
denoise_data = np.load('/home/sonic/Coding/Git/temp/Denoise_exp/clean_exp_data_24000.npy')
SAVE_MODEL_DIR = '/data2/torch_model_6th_4_4_model/'
POOL_PROCESS = 15
CUDA_DEVICE = 1
pool = Pool(processes=POOL_PROCESS)
torch.cuda.set_device(device=CUDA_DEVICE)

MAGNETIC_FIELD = 403.7139663551402            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000               # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32
PRE_PROCESSING = True
POWER = 12
def pre_processing(data, power=POWER):
    return data**power

SLOPE_INDEX = 11812
MEAN_PX_VALUE = 0.96     # exp_data[11755:11870].mean() + 0.018
slope = gaussian_slope(time_data, SLOPE_INDEX, MEAN_PX_VALUE, None) 
spin_bath = np.load("/home/sonic/Coding/Git/CPMG_Analysis/data/spin_bath_M_value.npy")

# filtered_idx 만드는 조건
MARGIN = 2              # peak에서 얼마나 넓게 볼것인지
WIDTH_THRESHOLD = 70    # peak 사이의 간격을 얼마나 잡을지.(이건 크게 영향을 미치지 않음)

# 모델을 만들 AB 범위
A_start = -60000
A_end = 60000
B_start = 2000
B_end = 74000
A_interval = 1000
B_interval = 2000

# 뽑을 spin 개수 및 target 스핀 한 개당 만들 data의 개수
n_of_spins = 28
n_of_small_spins = 13
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

# Model Hyperparameter
batch_size = 2048
learning_rate = 1e-4
epochs = 100

# Sample Number
N_VALID_SAMPLES = 2048
valid_data_batch = N_VALID_SAMPLES

A_pair1 = np.arange(A_start, A_end, A_interval)  
A_pair2 = np.arange(A_start, A_end, A_interval) + A_interval
A_range_list = [[a,b] for a,b in zip(A_pair1, A_pair2)]

B_pair1 = np.arange(B_start, B_end, B_interval)  
B_pair2 = np.arange(B_start, B_end, B_interval) + B_interval
B_range_list = [[a,b] for a,b in zip(B_pair1, B_pair2)]  

A = np.arange(0, len(A_range_list), 2)
B = np.arange(0, len(B_range_list), 2)

AB_indices = []
for A_temp in A:
    for B_temp in B:
        AB_indices.append([A_temp,B_temp])

model_step = 2  # model step
AB_total_dic = {}

for count, [i, j] in enumerate(AB_indices):
    A_test = A_range_list[i:i+model_step]
    B_test = B_range_list[j:j+model_step] 
    
    AB_dic = {}
    for idx1, A_range in enumerate(A_test):
        for idx2, B_range in enumerate(B_test):
            AB_dic[idx1*len(B_test)+idx2] = [[idx1, idx2], [A_range, B_range]] 
    AB_total_dic[count] = AB_dic
    
tic = time.time() # Time Caluation Start..

for idx_model, AB_dic in AB_total_dic.items():        
    if (idx_model <= 750) | (idx_model > 1100):
        continue
    if (idx_model % 18) < 3:
        train_data_batch = 4096 * 14
        PRE_PROCESSING = True
        POWER = 12
    else:
        train_data_batch = 4096 * 5
        PRE_PROCESSING = False
        
    AB_lists = np.zeros((len(AB_dic)+1, train_data_batch+valid_data_batch, n_of_spins, 2))
    total_filtered_idx = []
    
    for idx, AB_range in AB_dic.items():
        A_range = AB_range[1][0][0]
        B_range = AB_range[1][1][0]
        A = (A_range+A_interval/4) + A_interval/1.25 * np.random.uniform(0, 1, train_data_batch+valid_data_batch) 
        B = (B_range+B_interval/4) + B_interval/1.25 * np.random.uniform(0, 1, train_data_batch+valid_data_batch) 

        M_temp = M_list_return(time_data, WL_VALUE, [[(A_range+A_interval/2)*2*np.pi, (B_range+B_interval/2)*2*np.pi]], N_PULSE_32)
        fil_idx = get_filtered_idx(M_temp, margin=MARGIN, width_threshold=WIDTH_THRESHOLD)
        total_filtered_idx += list(fil_idx)

        AB_lists[idx, :, 0, :] = np.array([[A,B] for A,B in zip(A,B)])

        A_candidate = np.random.uniform(A_candidate_min, A_candidate_max, size=(train_data_batch+valid_data_batch, n_of_spins-n_of_small_spins-1))
        AB_lists[idx, :, 1:n_of_spins-n_of_small_spins, 0] = A_candidate
        A_small_candidate = np.random.uniform(A_small_min, A_small_max, size=(train_data_batch+valid_data_batch, n_of_small_spins))
        AB_lists[idx, :, n_of_spins-n_of_small_spins:, 0] = A_small_candidate

        B_candidate = np.random.uniform(B_candidate_min, B_candidate_max, size=(train_data_batch+valid_data_batch, n_of_spins-n_of_small_spins-1))
        AB_lists[idx, :, 1:n_of_spins-n_of_small_spins, 1] = B_candidate
        B_small_candidate = np.random.uniform(B_small_min, B_small_max, size=(train_data_batch+valid_data_batch, n_of_small_spins))
        AB_lists[idx, :, n_of_spins-n_of_small_spins:, 1] = B_small_candidate

        # 조건에 맞지않는 것 바꾸기
        bool_condition = (AB_lists[idx, :, 1:, 0] >= A_range) & (AB_lists[idx, :, 1:, 0] <= A_range+A_interval) \
                       & (AB_lists[idx, :, 1:, 1] >= B_range) & (AB_lists[idx, :, 1:, 1] <= B_range+B_interval)
        bool_mask = np.full(AB_lists.shape, False)
        bool_mask[idx, :, 1:, 0] = bool_condition
        bool_mask[idx, :, 1:, 1] = bool_condition

        count = 0
        model_indices = []
        while True in bool_condition:
            indices, _ = np.where(bool_mask[idx, :, 1:n_of_spins-n_of_small_spins, 0]==True)
            AB_lists[idx, :, 1:n_of_spins-n_of_small_spins, 0][bool_mask[idx, :, 1:n_of_spins-n_of_small_spins, 0]] \
                                    = np.random.uniform(A_candidate_min, A_candidate_max, size=indices.shape)
            AB_lists[idx, :, 1:n_of_spins-n_of_small_spins, 1][bool_mask[idx, :, 1:n_of_spins-n_of_small_spins, 0]] \
                                    = np.random.uniform(B_candidate_min, B_candidate_max, size=indices.shape)
            indices, _ = np.where(bool_mask[idx, :, n_of_spins-n_of_small_spins:, 0]==True)
            AB_lists[idx, :, n_of_spins-n_of_small_spins:, 0][bool_mask[idx, :, n_of_spins-n_of_small_spins:, 0]] \
                                    = np.random.uniform(A_small_min, A_small_max, size=indices.shape)
            AB_lists[idx, :, n_of_spins-n_of_small_spins:, 1][bool_mask[idx, :, n_of_spins-n_of_small_spins:, 1]] \
                                    = np.random.uniform(B_small_min, B_small_max, size=indices.shape)
            
            bool_condition = (AB_lists[idx, :, 1:, 0] >= A_range) & (AB_lists[idx, :, 1:, 0] <= A_range+A_interval) \
                           & (AB_lists[idx, :, 1:, 1] >= B_range) & (AB_lists[idx, :, 1:, 1] <= B_range+B_interval)
            bool_mask = np.full(AB_lists.shape, False)
            bool_mask[idx, :, 1:, 0] = bool_condition
            bool_mask[idx, :, 1:, 1] = bool_condition

    # 마지막 index의 A,B를 넣어주는 것(위에서는 target만 넣었으므로)
    AB_lists[-1, :, 0, 0] = np.random.uniform(A_candidate_min, A_candidate_max, train_data_batch+valid_data_batch)
    AB_lists[-1, :, 0, 1] = np.random.uniform(B_candidate_min, B_candidate_max, train_data_batch+valid_data_batch)

    bool_condition = ((AB_lists[-1, :, 0, 0] >= A_range) & (AB_lists[-1, :, 0, 0] <= A_range+A_interval)) & \
                      ((AB_lists[-1, :, 0, 1] >= B_range) & (AB_lists[-1, :, 0, 1] <= B_range+B_interval))
    while True in bool_condition:
        AB_lists[-1, :, 0, 0] = np.where(bool_condition,
                               np.random.uniform(A_candidate_min, A_candidate_max), AB_lists[-1, :, 0, 0]) 
        AB_lists[-1, :, 0, 1] = np.where(bool_condition, 
                               np.random.uniform(B_candidate_min, B_candidate_max), AB_lists[-1, :, 0, 1]) 
        bool_condition = ((AB_lists[-1, :, 0, 0] >= A_range) & (AB_lists[-1, :, 0, 0] <= A_range+A_interval)) & \
                          ((AB_lists[-1, :, 0, 1] >= B_range) & (AB_lists[-1, :, 0, 1] <= B_range+B_interval))

    total_filtered_idx = np.array(list(set(total_filtered_idx)))
    print("Model Index: ", idx_model, "total index length:", len(total_filtered_idx))
    
    ###############
    ### 2*np.pi 곱해주는 것은, 실수로 두번 곱하거나, 혹은 곱하지 않을 때가 많다. 반드시 확인해줘야 함. 확인하는 법은, X_train을 만들고 그래프의 value가 어떤지 확인할 것
    AB_lists = AB_lists*2*np.pi  

    #Generate training data
    for i in range(AB_lists.shape[0]):
        globals()["pool_{}".format(i)] = pool.apply_async(gen_X_train, [AB_lists[i][:train_data_batch], time_data[total_filtered_idx], WL_VALUE, N_PULSE_32])    
        globals()["pool_val_{}".format(i)] = pool.apply_async(gen_X_train, [AB_lists[i][train_data_batch:train_data_batch+valid_data_batch], time_data[total_filtered_idx], WL_VALUE, N_PULSE_32])    

    X_train_arr = np.zeros((AB_lists.shape[0], train_data_batch, total_filtered_idx.shape[0]))
    Y_train_arr = np.zeros((AB_lists.shape[0], X_train_arr.shape[1], AB_lists.shape[0]))
    X_valid_arr = np.zeros((AB_lists.shape[0], valid_data_batch, total_filtered_idx.shape[0]))
    Y_valid_arr = np.zeros((AB_lists.shape[0], X_valid_arr.shape[1], AB_lists.shape[0]))
    
    for j in range(AB_lists.shape[0]):
        X_train_arr[j] = globals()["pool_{}".format(j)].get(timeout=None)
        X_valid_arr[j] = globals()["pool_val_{}".format(j)].get(timeout=None)
        Y_train_arr[j, :, j] = 1
        Y_valid_arr[j, :, j] = 1

    print("Data Generation completed. Time consumed: {} (s)".format(time.time() - tic))
    print("===================== The AB_range of the model: ", AB_dic)
    print("(X, Y, X_val, Y_val) : ", X_train_arr.shape, Y_train_arr.shape, X_valid_arr.shape, Y_valid_arr.shape)

    X_train_arr = X_train_arr.reshape(-1, len(total_filtered_idx))
    Y_train_arr = Y_train_arr.reshape(-1, Y_train_arr.shape[-1])
    X_valid_arr = X_valid_arr.reshape(-1, len(total_filtered_idx))
    Y_valid_arr = Y_valid_arr.reshape(-1, Y_valid_arr.shape[-1])

    X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)
    X_valid_arr, Y_valid_arr = shuffle(X_valid_arr, Y_valid_arr)
    
    if PRE_PROCESSING == True:
        X_train_arr = pre_processing(X_train_arr, power=POWER)
        X_valid_arr = pre_processing(X_valid_arr, power=POWER)
        
    model = torch.nn.Sequential(
                nn.Linear(len(total_filtered_idx), 4096, bias=True),
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
                nn.Linear(512, 5, bias=True),
                nn.Sigmoid()
    ).cuda()
    summary(model, (1, len(total_filtered_idx)))
    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = len(X_train_arr)

    print("Training Start.. (Pre Process:{})".format(PRE_PROCESSING))
    model.train()
    total_val_loss = []
    total_acc = []

    for epoch in range(epochs):
        avg_cost = 0

        for i in range(total_batch // batch_size):
            indices = torch.randperm(total_batch)[:batch_size]
            x_train_temp = torch.tensor(X_train_arr[indices], dtype=torch.float).cuda()
            y_train_temp = torch.tensor(Y_train_arr[indices], dtype=torch.float).cuda()

            optimizer.zero_grad()
            hypothesis = model(x_train_temp)
            cost = criterion(hypothesis, y_train_temp)
            cost.backward()
            optimizer.step()

            avg_cost += cost

        print('Epoch:', '%4d' % (epoch + 1), ' | Loss =', '{:.5f}'.format(avg_cost / (total_batch // batch_size)), end=' | ')

        with torch.no_grad():
            model.eval()
            X_valid = torch.tensor(X_valid_arr, dtype=torch.float).cuda()
            Y_valid = torch.tensor(Y_valid_arr, dtype=torch.float).cuda()

            prediction = model(X_valid)
            val_loss = criterion(prediction, Y_valid)

            bool_pred = torch.argmax(prediction, dim=1, keepdim=True)
            bool_y = torch.argmax(Y_valid, dim=1, keepdim=True)
            # accuracy = torch.tensor(torch.sum(bool_pred == bool_y), dtype=torch.float) / len(X_valid_arr) * 100
            accuracy = torch.sum(bool_pred == bool_y).float() / len(X_valid_arr) * 100
            print('Val_loss: {:.5f} | Accuracy: {:.2f} %'.format(val_loss.item(), accuracy.item()))
            total_val_loss.append(val_loss.cpu().detach().item())
            total_acc.append(accuracy.cpu().detach().item())

        total_val_loss = np.array(total_val_loss)
        if total_acc[-1] > 92:
            break
        try:
            if np.min(total_val_loss[-12:-6]) < np.min(total_val_loss[-6:]):
                break
        except:pass
        total_val_loss = list(total_val_loss)

    toc =time.time()
    print("Training Done. Total Time Consumed: {} (s)".format(time.time()-tic))

    print("Prediction..")
    new_denoise = 2*denoise_data[:12000] - 1 
    new_denoise /= slope
    new_exp_data = 2*exp_data[:12000] - 1
    exp_data_unslope = new_exp_data / slope
    
    if PRE_PROCESSING == True:
        new_denoise = pre_processing(new_denoise, power=POWER)
        exp_data = pre_processing(new_exp_data, power=POWER)
        exp_data_unslope = pre_processing(exp_data_unslope, power=POWER)
        
    new_denoise_tensor = torch.FloatTensor(new_denoise[total_filtered_idx].reshape(1, -1)).cuda()
    exp_data_tensor = torch.FloatTensor(new_exp_data[total_filtered_idx].reshape(1, -1)).cuda()
    exp_data_unslope_tensor = torch.FloatTensor(exp_data_unslope[total_filtered_idx].reshape(1, -1)).cuda()

    pred_denoise = model(new_denoise_tensor)
    pred_exp = model(exp_data_tensor)
    pred_exp_unslope = model(exp_data_unslope_tensor)
    
    if os.path.isdir(SAVE_MODEL_DIR + "{}/".format(idx_model)):pass
    else:
        os.mkdir(SAVE_MODEL_DIR + "{}/".format(idx_model))
    
    np.save(SAVE_MODEL_DIR + "{}/{}_filtered_idx.npy".format(idx_model,idx_model), total_filtered_idx)
    np.save(SAVE_MODEL_DIR + "{}/{}_pred_denoise_values.npy".format(idx_model,idx_model), pred_denoise.cpu().detach().numpy())
    np.save(SAVE_MODEL_DIR + "{}/{}_pred_exp_values.npy".format(idx_model,idx_model), pred_exp.cpu().detach().numpy())
    np.save(SAVE_MODEL_DIR + "{}/{}_pred_exp_unslope.npy".format(idx_model,idx_model), pred_exp_unslope.cpu().detach().numpy())
    np.save(SAVE_MODEL_DIR + "{}/{}_val_loss_{}(s).npy".format(idx_model,idx_model, np.round(toc-tic)), total_val_loss)
    np.save(SAVE_MODEL_DIR + "{}/{}_val_acc.npy".format(idx_model,idx_model), np.array(total_acc))
    torch.save(model, SAVE_MODEL_DIR + '{}/{}_torch_model_{}'.format(idx_model, idx_model, PRE_PROCESSING))
    print("Model Saved.")

    del X_train_arr, X_valid_arr, model, X_valid, Y_valid
    torch.cuda.empty_cache()
    print("===================== Training Done. The AB_range of the model: ", AB_dic)
