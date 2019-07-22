from keras.layers import Dense, Activation, Dropout, Input, GaussianNoise, Conv1D, MaxPooling1D, Flatten, concatenate
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
N_SAMPLES_PER_PROCESS = 3000
GPU_INDEX = input("Enter the GPU index: ")
BATCH_SIZE = 4096

FILE_INDICES = np.array([0, 1, 2, 17, 18, 19, 34, 35, 36, 51, 52, 53, 68, 69, 70, 85, 86, 87, 102, 103, 104, 119, 120, 121, 136, 137, 138, 153, 154, 155, 
170, 171, 172, 187, 188, 189, 204, 205, 206, 221, 222, 223, 238, 239, 240, 255, 256, 257, 272, 273, 274, 289, 290, 291, 306, 307,
308, 323, 324, 325, 340, 341, 342, 357, 358, 359, 374, 375, 376, 391, 392, 393, 408, 409, 410, 425, 426, 427, 442, 443, 444, 459, 
460, 461, 476, 477, 478, 493, 494, 495, 510, 511, 512, 527, 528, 529, 544, 545, 546, 561, 562, 563, 578, 579, 580, 595, 596, 597, 
612, 613, 614, 629, 630, 631, 646, 647, 648, 663, 664, 665, 680, 681, 682, 697, 698, 699, 714, 715, 716, 731, 732, 733, 748, 749, 
750, 765, 766, 767, 782, 783, 784, 799, 800, 801, 816, 817, 818, 833, 834, 835, 850, 851, 852, 867, 868, 869, 884, 885, 886, 901,
902, 903, 918, 919, 920, 935, 936, 937, 952, 953, 954, 969, 970, 971, 986, 987, 988, 1003, 1004, 1005, 1020, 1021, 1022, 1037, 
1038, 1039, 1054, 1055, 1056, 1071, 1072, 1073, 1088, 1089, 1090, 1105, 1106, 1107, 1122, 1123, 1124, 1139, 1140, 1141, 1156,
1157, 1158, 1173, 1174, 1175, 1190, 1191, 1192, 1207, 1208, 1209, 1224, 1225, 1226, 1241, 1242, 1243, 1258, 1259, 1260, 1275, 
1276, 1277, 1292, 1293, 1294, 1309, 1310, 1311, 1326, 1327, 1328, 1343, 1344, 1345])
FILE_SLICE = np.arange(1,len(FILE_INDICES),3)

TIME_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_time.npy')*1e-6
EXP_DATA = np.load('/home/sonic/Coding/Git/temp/20190704_no_duplicated_expdata.npy')
time_data = TIME_DATA[:12000]
exp_data = EXP_DATA[:12000]
SLOPE_INDEX = 11575
MEAN_PX_VALUE = 0.97 #np.mean(EXP_DATA[11570:11580])
px_paper, slope = gaussian_slope(exp_data, time_data, SLOPE_INDEX, MEAN_PX_VALUE)

# START_IDX = int(input("Enter the start index : "))
# END_IDX = int(input("Enter the end index : ")) 
file_idx_dic = {}

for file_idx in FILE_INDICES[FILE_SLICE]:
    history = []
    total_time = []

    tic = time.time()
    file_list = glob.glob('/data1/keras_model/{}/{}_X*.npy'.format(file_idx, file_idx))
    file_list.sort()
    X_train = np.load(file_list[0])
    X_valid_32 = np.load(file_list[1])

    Y_train = np.zeros((len(X_train), 1))
    Y_valid = np.zeros((len(X_valid_32), 1))
    Y_train[:len(Y_train)//2 * 1] = np.ones((len(Y_train)//2, 1))
    Y_valid[:int(len(X_valid_32) / 2)] = np.ones((int(len(X_valid_32)/2), 1))
    
    X_train = (X_train + 1) / 2
    X_train = X_train * (1-X_train)
    X_train = 2*X_train-0.25
    for i in range(4, len(X_train)//2):
        temp = len(X_train.flatten()) // len(X_train)
        if temp % i == 0:
            X_train = X_train.reshape(len(X_train), i, -1)
            print("X_train_shape", X_train.shape)
            break

    X_valid_32 = (X_valid_32 + 1) / 2
    X_valid_32 = X_valid_32 * (1-X_valid_32)
    X_valid_32 = 2*X_valid_32-0.25
    X_valid_32 = X_valid_32.reshape(len(X_valid_32), X_train.shape[1], -1)

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    config = tf.ConfigProto(device_count = {'XLA_GPU' : 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    def gen_model(input_shape):
        X_train = Input(input_shape, dtype='float32')

        X = Conv1D(512, kernel_size = 2, activation='relu')(X_train)
        X = Conv1D(512, kernel_size = 2, activation='relu')(X)
        X = MaxPooling1D()(X)
        X = Flatten()(X)

        T = Conv1D(256, kernel_size = 2, activation='relu')(X_train)
        T = Conv1D(256, kernel_size = 2, activation='relu')(T)
        T = MaxPooling1D()(T)
        T = Flatten()(T)
        
        M = concatenate([X,T])
        M = GaussianNoise(0.01)(M)
        M = Dense(2048, activation='relu')(M)
        M = Dropout(0.3)(M)
        M = Dense(1024, activation='relu')(M)
        M = Dropout(0.2)(M)
        M = Dense(512, activation='relu')(M)
        M = Dropout(0.2)(M)

        X = Dense(1, activation='sigmoid')(M)

        model = Model(inputs=X_train, outputs=X)
        return model

    model = gen_model(input_shape)
    model.compile(optimizer=Adam(lr=0.8 * 1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    DATE = str(time.localtime().tm_year) + '_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday)
    save_directory = '/data1/keras_model_1dconv/{}/'.format(file_idx)
    if not(os.path.isdir(save_directory)):
        os.mkdir(save_directory)

    print(model.summary())
    current_time = time.localtime()[:]
    model_savename = str(current_time[0])+'_'+str(current_time[1])+'_'+str(current_time[2])+'_'+str(current_time[3])+'_'+str(current_time[4])+'_'
    checkpointer = ModelCheckpoint(save_directory + model_savename + 'model_'+str(file_idx)+'{epoch:02d}_{val_loss:.4f}.h5',
                                            save_best_only=True)

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    time_callback = TimeHistory()
    
    for i in range(100):
        history_temp = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                            epochs=8,  
                            callbacks=[checkpointer,time_callback], 
                            validation_data=(X_valid_32, Y_valid),
                            shuffle=True)
        
        time_temp = time_callback.times
        history.append(history_temp.history)
        if min(history_temp.history['val_loss']) < 0.1:break

    toc = time.time()
    total_time.append(toc-tic)
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

    filtered_idx_dic = np.load('/data1/keras_model/filter_indices.npy').item()
    exp_data2 = exp_data[filtered_idx_dic[file_idx]].reshape(1, -1)
    exp_data2 = exp_data2 * (1 - exp_data2)
    exp_data2 = 2*exp_data2 - 0.25
    y_pred = model.predict(exp_data2)
    try:
        heatmap = np.load('/data1/keras_model_1dconv/2nd_heatmap.npy')
    except:
        heatmap = np.zeros(len(FILE_INDICES), 11)
    heatmap[file_idx] = y_pred
    np.save('/data1/keras_model_1dconv/2nd_heatmap.npy', heatmap)
    
    filtered_arr_idx = np.array(filtered_idx_dic[file_idx])
    avg_heatmap_list = np.load('/data1/keras_model_1dconv/2nd_heatmap_2.npy')

    for i in range(-5, 6):
        temp_idx = filtered_arr_idx + i
        exp_data2 = exp_data[temp_idx].reshape(1, -1)
        exp_data2 = exp_data2 * (1 - exp_data2)
        exp_data2 = 2*exp_data2 - 0.25
        y_pred_temp = model.predict(exp_data2)
        y_pred_temp = y_pred_temp.reshape(1,)
        avg_heatmap_list[file_idx][i] = y_pred_temp[0]
    np.save('/data1/keras_model_1dconv/2nd_heatmap_2.npy', avg_heatmap_list)

    del X_train, X_valid_32, model
    
    sess.close()
    gc.collect()