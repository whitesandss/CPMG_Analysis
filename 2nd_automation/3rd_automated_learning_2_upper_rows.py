from keras.layers import Dense, Activation, Dropout, Input, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.models import load_model, Model
from keras import backend as K
from multiprocessing import Pool
import sys
sys.path.insert(0, '/home/sonic/Coding/Git/CPMG_Analysis/2nd_automation/')
from decom_utils import *
import numpy as np
import time
import os
import glob
import tensorflow as tf
import keras
import time
import gc

MAGNETIC_FIELD = 403.7139663551402            # Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.07*1000               # Unit: Herts
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
N_PULSE_32 = 32

N_FILES = 1
POOL_PROCESS = 23
N_SAMPLES_PER_PROCESS = 256
GPU_INDEX = '0'
BATCH_SIZE = 256
N_EPOCH = 20
MODEL_SAVE_DIR = '/data2/keras_model_3rd/'
# AB_LISTS_DIR = '/data2/AB_list_3rd/'
AB_LISTS_DIR = '/data/AB_list_3rd/'

TIME_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]
SLOPE_INDEX = 11575  # decoherence effect 기준 index
MEAN_PX_VALUE = 0.97 # np.mean(EXP_DATA[11570:11580])
px_paper, slope = gaussian_slope(exp_data, time_data, SLOPE_INDEX, MEAN_PX_VALUE)

# PRE_PRECESSING
PRE_PRECESSING = True

# filtered index 설정 범위
MARGIN = 5
WIDTH_THRESHOLD = 70

B_start = 2000
B_end = 70000
end_to_end_B_range = 2000
B_pair1 = np.arange(B_start, B_end, end_to_end_B_range)
B_pair2 = np.arange(B_start, B_end, end_to_end_B_range) + end_to_end_B_range
B_range_list = [[a,b] for a,b in zip(B_pair1, B_pair2)]

# 원하는 부분만 학습하고자 할 때
heatmap_n_rows = len(B_range_list)
target_indices = []
A_index_min = 70     # 원하는 A영역 index min, max
A_index_max = 70  
B_index_min = 25     # 원하는 B영역 index min, max
B_index_max = 30

                    # 0=64, 1=32, 2=16, 3=10, 4=4, 5=3, 6=2, 7이상=1 (AB_dic의 B인덱스=PRE_PRO_POWER)
PRE_PRO_POWER = 1   # 나중에 Prediction 할 때, 위의 값으로 반드시 알맞게 적용!!!
def pre_process(M_data, power=PRE_PRO_POWER):
    M_data = M_data**power
    return M_data

AB_dic = np.load('/data2/AB_dict.npy').item()
for i in range(len(AB_dic)):
    if (i//heatmap_n_rows >= A_index_min) & (i//heatmap_n_rows <= A_index_max):
        if (i%heatmap_n_rows >= B_index_min) & (i%heatmap_n_rows <= B_index_max):
            target_indices.append(i)

print(target_indices)

# START_IDX = int(input("Enter the start index : "))
# END_IDX = int(input("Enter the end index : ")) 
file_idx_dic = {}
go_collect_count = 0

for file_idx in target_indices:
    print("start_file_idx : ", file_idx)
    history = []
    total_time = []
    file_list = glob.glob(AB_LISTS_DIR + '{}_*.npy'.format(file_idx))
    file_list.sort()

    target_AB_included = np.load(file_list[1])
    none_target_AB = np.load(file_list[0])
    
    try:
        A_temp = int(file_list[1].split('_')[4].split('-')[0].split('(')[-1])
        B_temp = int(file_list[1].split('_')[5].split('-')[0].split('(')[-1])
    except:
        A_temp = -int(file_list[1].split('_')[4].split('-')[1])
        B_temp = int(file_list[1].split('_')[5].split('-')[0].split('(')[-1])

    A_temp = A_temp * 1000 + 500
    B_temp = B_temp * 1000 + 1000
    print("A:{}, B:{}".format(A_temp, B_temp))
    
    M_list = M_list_return(time_data, WL_VALUE, [[A_temp*2*np.pi, B_temp*2*np.pi]], N_PULSE_32)
    filtered_idx = get_filtered_idx(M_list, margin=MARGIN, width_threshold=WIDTH_THRESHOLD)
    filtered_time_data = time_data[filtered_idx]
    filetered_slope = slope[filtered_idx]
    file_idx_dic[file_idx] = filtered_idx
    
    def gen_AB_lists_to_px_lists(start_idx, target_AB_included, none_target_AB, filtered_time_data, count=1, n_samples=N_SAMPLES_PER_PROCESS):

        total_M_list = np.zeros((n_samples, len(filtered_time_data)))
        total_none_M_list = np.zeros((n_samples, len(filtered_time_data)))
        print("start_idx: ", start_idx)
        for i in range(n_samples):
            total_M_list[i] = M_list_return(filtered_time_data, WL_VALUE, target_AB_included[start_idx+i], N_PULSE_32)
            total_none_M_list[i] = M_list_return(filtered_time_data, WL_VALUE, none_target_AB[start_idx+i], N_PULSE_32)
        return total_M_list, total_none_M_list

    N_files=N_FILES
    PoolProcess=POOL_PROCESS
    init_n_samples=N_SAMPLES_PER_PROCESS
    count = 1
    start_n_samples = init_n_samples*25*(count-1)
            
    tic = time.time()
    pool = Pool(processes=PoolProcess)

    result1 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*0), target_AB_included, none_target_AB, filtered_time_data, count])
    result2 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*1), target_AB_included, none_target_AB, filtered_time_data, count])
    result3 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*2), target_AB_included, none_target_AB, filtered_time_data, count])
    result4 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*3), target_AB_included, none_target_AB, filtered_time_data, count])
    result5 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*4), target_AB_included, none_target_AB, filtered_time_data, count])
    result6 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*5), target_AB_included, none_target_AB, filtered_time_data, count])
    result7 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*6), target_AB_included, none_target_AB, filtered_time_data, count])
    result8 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*7), target_AB_included, none_target_AB, filtered_time_data, count])
    result9 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*8), target_AB_included, none_target_AB, filtered_time_data, count])
    result10 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*9), target_AB_included, none_target_AB, filtered_time_data, count])
    result11 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*10), target_AB_included, none_target_AB, filtered_time_data, count])
    result12 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*11), target_AB_included, none_target_AB, filtered_time_data, count])
    result13 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*12), target_AB_included, none_target_AB, filtered_time_data, count])
    result14 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*13), target_AB_included, none_target_AB, filtered_time_data, count])
    result15 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*14), target_AB_included, none_target_AB, filtered_time_data, count])
    result16 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*15), target_AB_included, none_target_AB, filtered_time_data, count])
    result17 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*16), target_AB_included, none_target_AB, filtered_time_data, count])
    result18 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*17), target_AB_included, none_target_AB, filtered_time_data, count])
    result19 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*18), target_AB_included, none_target_AB, filtered_time_data, count])
    result20 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*19), target_AB_included, none_target_AB, filtered_time_data, count])
    result21 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*20), target_AB_included, none_target_AB, filtered_time_data, count])
    result22 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*21), target_AB_included, none_target_AB, filtered_time_data, count])
    result23 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*22), target_AB_included, none_target_AB, filtered_time_data, count])
    if count==1:
        result24 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*23), target_AB_included, none_target_AB, filtered_time_data, count])
        result25 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*24), target_AB_included, none_target_AB, filtered_time_data, count])
        result26 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*25), target_AB_included, none_target_AB, filtered_time_data, count])
        result27 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*26), target_AB_included, none_target_AB, filtered_time_data, count])
        result28 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*27), target_AB_included, none_target_AB, filtered_time_data, count])
        result29 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*28), target_AB_included, none_target_AB, filtered_time_data, count])
        result29 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*29), target_AB_included, none_target_AB, filtered_time_data, count])
        result30 = pool.apply_async(gen_AB_lists_to_px_lists, [start_n_samples+(init_n_samples*30), target_AB_included, none_target_AB, filtered_time_data, count])

    X_train = np.zeros(((init_n_samples*46, len(filtered_time_data))))
    X_valid_32 = np.zeros(((init_n_samples*16, len(filtered_time_data))))
    # X_eval_32 = np.zeros(((init_n_samples*2, len(filtered_time_data))))

    X_train[init_n_samples*0:init_n_samples*1],   X_train[init_n_samples*23:init_n_samples*24] = result1.get(timeout=None)
    X_train[init_n_samples*1:init_n_samples*2],   X_train[init_n_samples*24:init_n_samples*25] = result2.get(timeout=None)
    X_train[init_n_samples*2:init_n_samples*3],   X_train[init_n_samples*25:init_n_samples*26] = result3.get(timeout=None)
    X_train[init_n_samples*3:init_n_samples*4],   X_train[init_n_samples*26:init_n_samples*27] = result4.get(timeout=None)
    X_train[init_n_samples*4:init_n_samples*5],   X_train[init_n_samples*27:init_n_samples*28] = result5.get(timeout=None)
    X_train[init_n_samples*5:init_n_samples*6],   X_train[init_n_samples*28:init_n_samples*29] = result6.get(timeout=None)
    X_train[init_n_samples*6:init_n_samples*7],   X_train[init_n_samples*29:init_n_samples*30] = result7.get(timeout=None)
    X_train[init_n_samples*7:init_n_samples*8],   X_train[init_n_samples*30:init_n_samples*31] = result8.get(timeout=None)
    X_train[init_n_samples*8:init_n_samples*9],   X_train[init_n_samples*31:init_n_samples*32] = result9.get(timeout=None)
    X_train[init_n_samples*9:init_n_samples*10],  X_train[init_n_samples*32:init_n_samples*33] = result10.get(timeout=None)
    X_train[init_n_samples*10:init_n_samples*11], X_train[init_n_samples*33:init_n_samples*34] = result11.get(timeout=None)
    X_train[init_n_samples*11:init_n_samples*12],   X_train[init_n_samples*34:init_n_samples*35] = result12.get(timeout=None)
    X_train[init_n_samples*12:init_n_samples*13],   X_train[init_n_samples*35:init_n_samples*36] = result13.get(timeout=None)
    X_train[init_n_samples*13:init_n_samples*14],   X_train[init_n_samples*36:init_n_samples*37] = result14.get(timeout=None)
    X_train[init_n_samples*14:init_n_samples*15],   X_train[init_n_samples*37:init_n_samples*38] = result15.get(timeout=None)
    X_train[init_n_samples*15:init_n_samples*16],   X_train[init_n_samples*38:init_n_samples*39] = result16.get(timeout=None)
    X_train[init_n_samples*16:init_n_samples*17],   X_train[init_n_samples*39:init_n_samples*40] = result17.get(timeout=None)
    X_train[init_n_samples*17:init_n_samples*18],   X_train[init_n_samples*40:init_n_samples*41] = result18.get(timeout=None)
    X_train[init_n_samples*18:init_n_samples*19],   X_train[init_n_samples*41:init_n_samples*42] = result19.get(timeout=None)
    X_train[init_n_samples*19:init_n_samples*20],   X_train[init_n_samples*42:init_n_samples*43] = result20.get(timeout=None)
    X_train[init_n_samples*20:init_n_samples*21],   X_train[init_n_samples*43:init_n_samples*44] = result21.get(timeout=None)
    X_train[init_n_samples*21:init_n_samples*22],   X_train[init_n_samples*44:init_n_samples*45] = result22.get(timeout=None)
    X_train[init_n_samples*22:init_n_samples*23],   X_train[init_n_samples*45:init_n_samples*46] = result23.get(timeout=None)

    if count==1:
        X_valid_32[init_n_samples*0:init_n_samples*1], X_valid_32[init_n_samples*8:init_n_samples*9] = result24.get(timeout=None)
        X_valid_32[init_n_samples*1:init_n_samples*2], X_valid_32[init_n_samples*9:init_n_samples*10] = result24.get(timeout=None)
        X_valid_32[init_n_samples*2:init_n_samples*3], X_valid_32[init_n_samples*10:init_n_samples*11] = result24.get(timeout=None)
        X_valid_32[init_n_samples*3:init_n_samples*4], X_valid_32[init_n_samples*11:init_n_samples*12] = result24.get(timeout=None)
        X_valid_32[init_n_samples*4:init_n_samples*5], X_valid_32[init_n_samples*12:init_n_samples*13] = result24.get(timeout=None)
        X_valid_32[init_n_samples*5:init_n_samples*6], X_valid_32[init_n_samples*13:init_n_samples*14] = result24.get(timeout=None)
        X_valid_32[init_n_samples*6:init_n_samples*7], X_valid_32[init_n_samples*14:init_n_samples*15] = result24.get(timeout=None)
        X_valid_32[init_n_samples*7:init_n_samples*8], X_valid_32[init_n_samples*15:init_n_samples*16] = result24.get(timeout=None)
        # X_eval_32[init_n_samples*0:init_n_samples*1], X_eval_32[init_n_samples*1:init_n_samples*2] = result25.get(timeout=None)

    pool.close()
    pool.join()
    toc = time.time()
    print("Calculated Time : {} s".format(toc-tic))
    print("X_train_size : ", getsizeof_variable(X_train))
    
    Y_train = np.zeros((len(X_train), 1))
    Y_valid = np.zeros((len(X_valid_32), 1))
    # Y_eval = np.zeros((len(X_eval_32), 1))
    Y_train[:len(Y_train)//2 * 1] = np.ones((len(Y_train)//2, 1))
    Y_valid[:int(len(X_valid_32) / 2)] = np.ones((int(len(X_valid_32)/2), 1))
    # Y_eval[:int(len(X_eval_32) / 2)] = np.ones((int(len(X_eval_32)/2), 1))

    X_train *= filetered_slope
    X_valid_32 *= filetered_slope

    if PRE_PRECESSING:
        X_train = pre_process(X_train, power=PRE_PRO_POWER)
        X_valid_32 = pre_process(X_valid_32, power=PRE_PRO_POWER)
#         X_train = (X_train + 1) / 2
#         X_train = X_train * (1-X_train)
#         X_train = X_train**PRE_PRO_POWER
#         X_train = ((2*X_train)**8) - 1
#         X_valid_32 = (X_valid_32 + 1) / 2
#         X_valid_32 = X_valid_32 * (1-X_valid_32)
#         X_valid_32 = X_valid_32**PRE_PRO_POWER
#         X_valid_32 = ((2*X_valid_32)**8) - 1
    
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    config = tf.ConfigProto(device_count = {'XLA_GPU' : 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    input_shape = (len(filtered_time_data),)

    def gen_model(input_shape):
        X_train = Input(input_shape, dtype='float32')
        # X = Dense(4096, activation='relu')(X_train)
        # X = Dropout(0.4)(X)
        X = Dense(2048, activation='relu')(X_train)
        # X = Dense(2048, activation='relu')(X)
        X = Dropout(0.3)(X)
        X = Dense(1024, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(512, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=X_train, outputs=X)
        return model

    model = gen_model(input_shape)
    model.compile(optimizer=Adam(lr=0.8 * 1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    DATE = str(time.localtime().tm_year) + '_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday)
    save_directory = MODEL_SAVE_DIR + '{}/'.format(file_idx)
            
    if os.path.isdir(save_directory + '{}/'.format(file_idx)):
        pass
    else:
        try:
            os.mkdir(save_directory)
        except:
            pass
    
    current_time = time.localtime()[:]
    model_savename = str(current_time[0])+'_'+str(current_time[1])+'_'+str(current_time[2])+'_'+str(current_time[3])+'_'+str(current_time[4])+'_'
    checkpointer = ModelCheckpoint(save_directory + model_savename + 'model_{}_{}_{}.h5'.format(file_idx, A_temp, B_temp),
                                            save_best_only=True)

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    time_callback = TimeHistory()
    
    history_temp = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                        epochs=N_EPOCH,
                        callbacks=[checkpointer,time_callback], 
                        validation_data=(X_valid_32, Y_valid),
                        shuffle=True)

    time_temp = time_callback.times
    history.append(history_temp.history)
    total_time.append(time_temp)
#     if min(history_temp.history['val_loss']) < 0.1:break
    
    np.save(save_directory + "{}_history.npy".format(file_idx), history)
    np.save(save_directory + "{}_time.npy".format(file_idx), total_time)
    list_of_save_files = glob.glob(save_directory+'*.h5')
    try:
        latest_file = max(list_of_save_files, key=os.path.getctime)
        list_of_save_files.remove(latest_file)
        for file in list_of_save_files: os.remove(file)
    except:
        pass
    
    model = load_model(latest_file)

    if PRE_PRECESSING:
        exp_data2 = exp_data[filtered_idx].reshape(1, -1)
        exp_data2 = 2*exp_data2 - 1
        exp_data2 = pre_process(exp_data2, power=PRE_PRO_POWER)
#         exp_data2 = exp_data2**PRE_PRO_POWER
#         exp_data2 = (2*exp_data2**8) - 1
    else:
        exp_data2 = exp_data[filtered_idx].reshape(1, -1)
        exp_data2 = 2*exp_data2 - 1

    y_pred = model.predict(exp_data2)

    heatmap = np.load(MODEL_SAVE_DIR + '3rd_heatmap.npy')
    idx1 = AB_dic[file_idx][0][0]
    idx2 = AB_dic[file_idx][0][1]
    heatmap[idx1,idx2] = y_pred
    np.save(MODEL_SAVE_DIR + '3rd_heatmap.npy', heatmap)
    
    filtered_arr_idx = np.array(filtered_idx)
    avg_heatmap_list = np.load(MODEL_SAVE_DIR + '3rd_heatmap_shift.npy')

    for i in range(-3, 4):
        temp_idx = filtered_arr_idx + i
        y_pred_temp = model.predict(exp_data2)
        y_pred_temp = y_pred_temp.reshape(1,)
        avg_heatmap_list[idx1,idx2][i] = y_pred_temp[0]
    np.save(MODEL_SAVE_DIR + '3rd_heatmap_shift.npy', avg_heatmap_list)
    np.save(save_directory + '{}_filtered_idx.npy'.format(file_idx), filtered_idx)
    print("save completed")
#     del X_train, X_valid_32
    
    sess.close()
    gc.collect()