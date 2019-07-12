# import modules
import numpy as np
import os, sys, glob
from decom_utils.py import *
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import random
import time
import pickle
from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, CuDNNLSTM, Reshape
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.models import load_model, Model
from keras import backend as K

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

def do_train(A_idx, B_idx, loss_threshold = 0.15, GPU_index = '1', Preprocess = False):
    '''
    function do trains! 
    Parameter: AB_range_index   -> formatted string with format A_index-B_index. For example, 0-0 or 0-1 should be entered.
    '''

    A_range = A_range_list[A_idx]
    B_range = B_range_list[B_idx]
    AB_range_index = str(A_idx) + '-' + str(B_idx)

    A_min = A_range[0]
    A_max = A_range[1]
    B_min = B_range[0]
    B_max = B_range[1]

    AB_range = 'A(' + str(A_min//1000) +'-'+ str(A_max//1000) + ')_B(' + str(B_min//1000) +'-'+ str(B_max//1000) + ')'

    X_valid_32 = np.load('/media/sonic/Samsung_T5/CPMG_data/{}/{}_valid_px_list_'.format(AB_range_index, AB_range_index) + AB_range + '_7spin.npy')
    X_eval_32 = np.load('/media/sonic/Samsung_T5/CPMG_data/{}/{}_eval_px_list_'.format(AB_range_index, AB_range_index) + AB_range + '_7spin.npy')
    if X_valid_32.shape != (2800, 12000):
        raise ValueError("Valid data Shape does not match\nCheck file {}_valid_px_list_".format(AB_range_index) + AB_range + '_7spin.npy')
    if X_eval_32.shape != (2800, 12000):
        raise ValueError("Eval data Shape does not match\nCheck file {}_eval_px_list_".format(AB_range_index) + AB_range + '_7spin.npy')

    # pre-processing 
    if Preprocess:
        X_valid_32 = (X_valid_32 + 1) / 2
        X_eval_32 = (X_eval_32 + 1) / 2
        X_valid_32 = X_valid_32 * (1-X_valid_32)
        X_eval_32 = X_eval_32 * (1-X_eval_32)
        X_valid_32 = 2*X_valid_32-0.25
        X_eval_32 = 2*X_eval_32-0.25
    
    Y_valid = np.zeros((len(X_valid_32),1))
    Y_eval = np.zeros((len(X_eval_32),1))

    Y_valid[:int(len(X_valid_32)/2)] = np.ones((int(len(X_valid_32)/2), 1))
    Y_eval[:int(len(X_eval_32)/2)] = np.ones((int(len(X_eval_32)/2), 1))

    store_directory = '/media/sonic/Samsung_T5/CPMG_data/{}/'.format(AB_range_index)
    file_list = os.listdir(store_directory)
    train_file_lists = []
    for file_name in file_list:
        if file_name.find('train_px_list_'+AB_range) != -1:
            train_file_lists.append(file_name)
    train_file_lists.sort()
    print(train_file_lists)

    def train_gen(batch_size):
        train_index = np.random.randint(len(train_file_lists))
        X_train = np.load(store_directory + train_file_lists[train_index])
        if Preprocess:
            X_train = (X_train + 1) / 2
            X_train = X_train * (1 - X_train)
            X_train = 2 * X_train - 0.25
        
        Y_train = np.zeros((len(X_train), 1))
        Y_train[: len(Y_train)//2 * 1] = np.ones((len(Y_train)//2, 1))
        indices = np.arange(0, len(X_train) - batch_size)
        
        while 1:
            start_idx = np.random.choice(indices)
            x_train = X_train[start_idx:start_idx + batch_size]
            y_train = Y_train[start_idx:start_idx + batch_size]
            yield x_train, y_train

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index
    except:
        raise ValueError("Please check the GPU index")

    config = tf.ConfigProto(device_count = {'XLA_GPU' : 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    def model_gen(input_shape):
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
        
    input_shape = (12000,)
    model = model_gen(input_shape)
    # print(model.summary())
    model.compile(optimizer=Adam(lr=0.8 * 1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
    DATE = str(time.localtime().tm_year) + '_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday)
    if Preprocess:
        save_directory = '/home/sonic/Coding/Decomposition/Simul_Data/keras_save_model/{}_Dense_Tim_classify_N32_12000_'.format(DATE)+AB_range_index+'_'+AB_range+Preprocess+'/'    
    else:
        save_directory = '/home/sonic/Coding/Decomposition/Simul_Data/keras_save_model/{}_Dense_Tim_classify_N32_12000_'.format(DATE)+AB_range_index+'_'+AB_range+'/'

    if not(os.path.isdir(save_directory)):
        os.mkdir(save_directory)

    current_time = time.localtime()[:]
    model_savename = str(current_time[0])+'_'+str(current_time[1])+'_'+str(current_time[2])+'_'+str(current_time[3])+'_'+str(current_time[4])+'_'
    checkpointer = ModelCheckpoint(save_directory + model_savename + 'model_{epoch:2d}_{val_loss:5.5f}.h5', 
                               save_best_only=True)

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    time_callback = TimeHistory()

    tp_loss = []
    history = []
    total_time = []
    max_iter = 25
    N_iter = 0
    for i in range(max_iter):
        history_temp = model.fit_generator(train_gen(batch_size=256),
                                steps_per_epoch=128, 
                                epochs=4, 
                                callbacks=[checkpointer,time_callback], 
                                validation_data=(X_valid_32, Y_valid),
                                shuffle=True)
        time_temp = time_callback.times
        history.append(history_temp)
        total_time.append(time_temp)
        print("{} Iteration Completed.".format(i+1))

        list_of_save_files = glob.glob(save_directory+'*.h5')
        latest_file = max(list_of_save_files, key=os.path.getctime)
        list_of_save_files.remove(latest_file)
        for file in list_of_save_files: os.remove(file)

        if (np.array(history_temp.history['val_loss']) < loss_threshold).sum() >=2 :
            N_iter = i + 1
            break
    
    time_table = np.load('/home/sonic/Coding/Decomposition/TimTaminiau/20190610_time_no_duplicated.npy')[:6000]*1e-6
    exp_data_tim = np.load('/home/sonic/Coding/Decomposition/TimTaminiau/20190610_exp_data_24200.npy')[:15000]
    y_pred = []
    for i in range(10):
        exp_data_temp = exp_data_tim[i:i+12000]
        exp_data_temp = exp_data_temp.transpose()
        if Preprocess:
            exp_data_temp = exp_data_temp * (1-exp_data_temp)
            exp_data_temp = 2*exp_data_temp-0.25
        else:
            exp_data_temp = exp_data_temp * 2 - 1
        y_pred.append(model.predict(exp_data_temp))
    print("predicted values:")
    print(y_pred)
    with open('/home/sonic/Coding/Decomposition/Simul_Data/history/{}'.format(AB_range_index)+'.bin', 'wb') as hist_f:
        pickle.dump(history, hist_f)

    # save logs
    with open('/home/sonic/Coding/Decomposition/Simul_Data/log.txt', 'a') as f:
        f.write("*********************************************************************" + "\n")
        f.write("Index: " +  AB_range_index + "AB_range: " + AB_range + "\n")
        f.write("With " + str(N_iter) + " iteration, " + "\n")
        f.write("loss: " + str(history[-1].history['loss'][-1]) + "\n")
        f.write("acc: " + str(history[-1].history['acc'][-1]) + "\n")
        f.write("val_loss: " + str(history[-1].history['val_loss'][-1]) + "\n")
        f.write("val_acc: " + str(history[-1].history['val_acc'][-1]) + "\n")
        f.write("predicted values: \n")
        for line in y_pred:
            f.write(str(line.tolist()))
    data_folder = '/media/sonic/Samsung_T5/CPMG_data/{}/'.format(AB_range_index)
    list_of_files = glob.glob(data_folder+'*.npy')
    for file in list_of_files: os.remove(file)

    return y_pred
    
if __name__ == "__main__":
    pred = do_train(0, 0, loss_threshold = 0.15, GPU_index = '1', Preprocess = 'False')
    