from keras.layers import Dense, Activation, Dropout, Input, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.models import load_model, Model
from keras import backend as K
from multiprocessing import Pool
import sys
sys.path.insert(0, '/home/sonic/Coding/Git/CPMG_Analysis/')
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
N_SAMPLES_PER_PROCESS = 8192
GPU_INDEX = '1'
BATCH_SIZE = 4096

a = [0, 1, 2, 17, 18, 19, 34, 35, 36, 51, 52, 53, 68, 69, 70, 85, 86, 87, 102, 103, 104, 119, 120, 121, 136, 137, 138, 153, 154, 155, 
170, 171, 172, 187, 188, 189, 204, 205, 206, 221, 222, 223, 238, 239, 240, 255, 256, 257, 272, 273, 274, 289, 290, 291, 306, 307, 
308, 323, 324, 325, 340, 341, 342, 357, 358, 359, 374, 375, 376, 391, 392, 393, 408, 409, 410, 425, 426, 427, 442, 443, 444, 459,
460, 461, 476, 477, 478, 493, 494, 495, 510, 511, 512, 527, 528, 529, 544, 545, 546, 561, 562, 563, 578, 579, 580, 595, 596, 597, 
612, 613, 614, 629, 630, 631, 646, 647, 648, 663, 664, 665, 680, 681, 682, 697, 698, 699, 714, 715, 716, 731, 732, 733, 748, 749]

# a = [750, 765, 766, 767, 782, 783, 784, 799, 800, 801, 816, 817, 818, 833, 834, 835, 850, 851, 852, 867, 868, 869, 884, 885, 886, 901,
# 902, 903, 918, 919, 920, 935, 936, 937, 952, 953, 954, 969, 970, 971, 986, 987, 988, 1003, 1004, 1005, 1020, 1021, 1022, 1037, 
# 1038, 1039, 1054, 1055, 1056, 1071, 1072, 1073, 1088, 1089, 1090, 1105, 1106, 1107, 1122, 1123, 1124, 1139, 1140, 1141, 1156,
# 1157, 1158, 1173, 1174, 1175, 1190, 1191, 1192, 1207, 1208, 1209, 1224, 1225, 1226, 1241, 1242, 1243, 1258, 1259, 1260, 1275, 
# 1276, 1277, 1292, 1293, 1294, 1309, 1310, 1311, 1326, 1327, 1328, 1343, 1344, 1345]

TIME_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]
SLOPE_INDEX = 11575
MEAN_PX_VALUE = 0.97 #np.mean(EXP_DATA[11570:11580])
px_paper, slope = gaussian_slope(exp_data, time_data, SLOPE_INDEX, MEAN_PX_VALUE)

START_IDX = int(input("Enter the start index : "))
END_IDX = int(input("Enter the end index : ")) 
file_idx_dic = {}
for file_idx in a:
    print("file_idx", file_idx)
    history = []
    total_time = []
    file_list = glob.glob('/home/sonic/Coding/Decomposition/Simul_Data/AB_lists/{}_*.npy'.format(file_idx))
    file_list.sort()
    target_AB_included = np.load(file_list[1])
    none_target_AB = np.load(file_list[0])

    try:
        A_temp = int(file_list[1].split('_')[4].split('-')[0].split('(')[-1])
        B_temp = int(file_list[1].split('_')[5].split('-')[0].split('(')[-1])
    except:
        A_temp = -int(file_list[1].split('_')[4].split('-')[1])
        B_temp = int(file_list[1].split('_')[5].split('-')[0].split('(')[-1])
    
    if A_temp % 3 == 0:
        A_temp = A_temp * 1000 + 750
        B_temp = B_temp * 1000 + 2000
        print("A:{}, B:{}".format(A_temp, B_temp))
    else:
        A_temp = A_temp * 1000 + 1250
        B_temp = B_temp * 1000 + 2000
        print("A:{}, B:{}".format(A_temp, B_temp))
        
    M_list = M_list_return(time_data, WL_VALUE, [[A_temp*2*np.pi, B_temp*2*np.pi]], N_PULSE_32)
    filtered_idx = get_filtered_idx(M_list)
    filtered_time_data = time_data[filtered_idx]
    filetered_slope = slope[filtered_idx]
    file_idx_dic[file_idx] = filtered_idx

    def gen_AB_lists_to_px_lists(start_idx, target_AB_included, none_target_AB, filtered_time_data, count=1, n_samples=N_SAMPLES_PER_PROCESS):

        X_train = np.zeros((n_samples, len(filtered_time_data)))
        total_none_M_list = np.zeros((n_samples, len(filtered_time_data)))
        print("start_idx: ", start_idx)
        for i in range(n_samples):
            X_train[i] = M_list_return(filtered_time_data, WL_VALUE, target_AB_included[start_idx+i], N_PULSE_32) * filetered_slope
            total_none_M_list[i] = M_list_return(filtered_time_data, WL_VALUE, none_target_AB[start_idx+i], N_PULSE_32) * filetered_slope
        return X_train, total_none_M_list

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

    X_train = np.zeros(((init_n_samples*46, len(filtered_time_data))))
    X_valid_32 = np.zeros(((init_n_samples*2, len(filtered_time_data))))
    X_eval_32 = np.zeros(((init_n_samples*2, len(filtered_time_data))))

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
        X_valid_32[init_n_samples*0:init_n_samples*1], X_valid_32[init_n_samples*1:init_n_samples*2] = result24.get(timeout=None)
        # X_eval_32[init_n_samples*0:init_n_samples*1], X_eval_32[init_n_samples*1:init_n_samples*2] = result25.get(timeout=None)

    pool.close()
    pool.join()
    toc = time.time()
    print("Calculated Time : {} s".format(toc-tic))
    
    Y_train = np.zeros((len(X_train), 1))
    Y_valid = np.zeros((len(X_valid_32), 1))
    # Y_eval = np.zeros((len(X_eval_32), 1))
    Y_train[:len(Y_train)//2 * 1] = np.ones((len(Y_train)//2, 1))
    Y_valid[:int(len(X_valid_32) / 2)] = np.ones((int(len(X_valid_32)/2), 1))
    # Y_eval[:int(len(X_eval_32) / 2)] = np.ones((int(len(X_eval_32)/2), 1))
    
    np.save('/data1/keras_model/{}/{}_X_train.npy'.format(file_idx,file_idx),X_train)
    np.save('/data1/keras_model/{}/{}_X_valid.npy'.format(file_idx,file_idx),X_valid_32)
'''
    X_train = (X_train + 1) / 2
    X_train = X_train * (1-X_train)
    X_train = 2*X_train-0.25
    X_valid_32 = (X_valid_32 + 1) / 2
    X_valid_32 = X_valid_32 * (1-X_valid_32)
    X_valid_32 = 2*X_valid_32-0.25
    
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    config = tf.ConfigProto(device_count = {'XLA_GPU' : 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    input_shape = (len(filtered_time_data),)

    def gen_model(input_shape):
        X_train = Input(input_shape, dtype='float32')
        X = Dense(4096, activation='relu')(X_train)
        X = Dropout(0.4)(X)
        X = Dense(2048, activation='relu')(X)
        X = Dropout(0.2)(X)
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
    save_directory = '/data1/keras_model/{}/'.format(file_idx)
    if not(os.path.isdir(save_directory)):
        os.mkdir(save_directory)

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
    
    for i in range(10):
        history_temp = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                            epochs=8,  
                            callbacks=[checkpointer,time_callback], 
                            validation_data=(X_valid_32, Y_valid),
                            shuffle=True)
        
        time_temp = time_callback.times
        history.append(history_temp.history)
        total_time.append(time_temp)
        if min(history_temp.history['val_loss']) < 0.1:break
    
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
    exp_data2 = exp_data[filtered_idx].reshape(1, -1)
    exp_data2 = exp_data2 * (1 - exp_data2)
    exp_data2 = 2*exp_data2 - 0.25
    y_pred = model.predict(exp_data2)
    heatmap = np.load('/data1/keras_model/2nd_heatmap.npy')
    heatmap[file_idx] = y_pred
    np.save('/data1/keras_model/2nd_heatmap.npy', heatmap)
    
    filtered_arr_idx = np.array(filtered_idx)
    avg_heatmap_list = np.load('/data1/keras_model/2nd_heatmap_2.npy')

    for i in range(-5, 6):
        temp_idx = filtered_arr_idx + i
        exp_data2 = exp_data[temp_idx].reshape(1, -1)
        exp_data2 = exp_data2 * (1 - exp_data2)
        exp_data2 = 2*exp_data2 - 0.25
        y_pred_temp = model.predict(exp_data2)
        y_pred_temp = y_pred_temp.reshape(1,)
        avg_heatmap_list[file_idx][i] = y_pred_temp[0]
    np.save('/data1/keras_model/2nd_heatmap_2.npy', avg_heatmap_list)

    del X_train, X_valid_32
    
    # keras.backend.clear_session()
    sess.close()
    gc.collect() 
    # os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    # config = tf.ConfigProto(device_count = {'XLA_GPU' : 0})
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # keras.backend.set_session(sess)
'''