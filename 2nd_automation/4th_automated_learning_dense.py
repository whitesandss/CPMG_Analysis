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

TIME_DATA = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/20190610_time_no_duplicated.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/CPMG_Analysis/data/20190610_exp_data_24200.npy').squeeze()
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]
exp_data = 2*exp_data - 1
MAGNETIC_FIELD = 403.7139663551402            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000               # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32
SAVE_MODEL_DIR = '/data2/torch_model_4th/'
POOL_PROCESS = 15
CUDA_DEVICE = 1

data_batch = 40960
valid_data_batch = 1024
batch_size = 4096
learning_rate = 1e-3
epochs = 100
pool = Pool(processes=POOL_PROCESS)
torch.cuda.set_device(device=CUDA_DEVICE)

file_list = glob.glob('/home/sonic/Coding/Git/ABlists_4th_model/total_*.npy')
file_list.sort()

def pre_processing(data, power=8):
    return data**power

# for idx in range(len(file_list)//2//3 + len(file_list)//2//3, len(file_list)//2, 2):
# for idx in range(len(file_list)//2//3 + 1, len(file_list)//2//3 + len(file_list)//2//3, 2):
# for idx in range(len(file_list)//2//3 + 1, 2):
# for idx in range(0, len(file_list)//2 + 1, 2):
for idx in range(len(file_list)//2, len(file_list), 2):
    
    X_data = np.load(file_list[idx])
    filtered_indices = np.load(file_list[idx+1])

    print("Generate data from the file: {}_{}".format(idx, file_list[idx].split('total_')[-1].split('.npy')[0]))
    tic = time.time()

    PRE_PROCESSING = False
    B_limit = int(file_list[idx].split('total_')[1].split('_')[-1].split('.')[0])

    if (B_limit > 4000) & (B_limit < 20000):
        power = 8
        PRE_PROCESSING = True
        print("PRE_PRECESSING Changed: False --> True, power:", power)

    if (B_limit > 24000):
        data_batch = 10240 // 2 
        print("data_batch Changed: ", data_batch)

    for i in range(X_data.shape[0]):
        globals()["pool_{}".format(i)] = pool.apply_async(gen_X_train, [X_data[i][:data_batch], time_data[filtered_indices], WL_VALUE, N_PULSE_32])    
        globals()["pool_val_{}".format(i)] = pool.apply_async(gen_X_train, [X_data[i][data_batch:data_batch+valid_data_batch], time_data[filtered_indices], WL_VALUE, N_PULSE_32])    

    X_train_arr = np.zeros((X_data.shape[0], data_batch, filtered_indices.shape[0]))
    X_valid_arr = np.zeros((X_data.shape[0], valid_data_batch, filtered_indices.shape[0]))
    
    for j in range(10):
        X_train_arr[j] = globals()["pool_{}".format(j)].get(timeout=None)
        X_valid_arr[j] = globals()["pool_val_{}".format(j)].get(timeout=None)
        print(j, end=", ")

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

    if PRE_PROCESSING==True:
        X_train_arr = pre_processing(X_train_arr, power=power)
        X_valid_arr = pre_processing(X_valid_arr, power=power)

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
    ).cuda()

    summary(model, (1, len(filtered_indices)))
    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = len(X_train_arr)

    tic =time.time()
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

        if total_acc[-1] > 92:
            break
        try:
            if ((total_val_loss[-8] + total_val_loss[-7] + total_val_loss[-6] + total_val_loss[-5]) - \
                max(total_val_loss[-8], total_val_loss[-7], total_val_loss[-6], total_val_loss[-5])) < \
               ((total_val_loss[-4] + total_val_loss[-3] + total_val_loss[-2] + total_val_loss[-1]) - \
                max(total_val_loss[-4], total_val_loss[-3], total_val_loss[-2], total_val_loss[-1])):
                break
        except:pass

    toc =time.time()
    print("Training Done. {} (s)".format(toc-tic))

    print("Prediction..")

    if PRE_PROCESSING:
        exp_data = pre_processing(exp_data, power=power)
    else:
        pass
        
    exp_tensor = torch.FloatTensor(exp_data[filtered_indices].reshape(1, -1)).cuda()
    pred = model(exp_tensor)

    folder_idx = int(file_list[idx].split('total_')[-1].split('.npy')[0].split('_')[1])

    if os.path.isdir(SAVE_MODEL_DIR + "{}/".format(folder_idx)):pass
    else:
        os.mkdir(SAVE_MODEL_DIR + "{}/".format(folder_idx))
    
    np.save(SAVE_MODEL_DIR + "{}/{}_pred_values.npy".format(folder_idx,folder_idx), pred.cpu().detach().numpy())
    np.save(SAVE_MODEL_DIR + "{}/{}_val_loss_{}(s).npy".format(folder_idx,folder_idx, np.round(toc-tic)), np.array(total_val_loss))
    np.save(SAVE_MODEL_DIR + "{}/{}_val_acc.npy".format(folder_idx,folder_idx), np.array(total_acc))
    torch.save(model, SAVE_MODEL_DIR + '{}/{}_torch_model_{}'.format(folder_idx, folder_idx, PRE_PROCESSING))
    print("Model Saved.")

    del X_train_arr, X_valid_arr, model, X_valid, Y_valid
    torch.cuda.empty_cache()
    print("===================== Training Done. File: {}".format(file_list[idx].split('total_')[-1]))