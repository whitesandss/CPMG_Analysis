import numpy as np
from automation import data_generator
from automation import automate_learning

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

for j in range(6, len(B_range_list), 2):
    for i in range(0, len(A_range_list), 2):
        print("A index is " + str(i) + "and B index is " + str(j))
        data_generator.generate(A_range_list[i], i, B_range_list[j], j, N_files = 3, PoolProcess=23)
        pred = automate_learning.do_train(i, j, loss_threshold = 0.15, GPU_index = '0', Preprocess = 'False')
        heatmap = np.load("/home/sonic/Coding/Decomposition/Simul_Data/heatmap.npy")
        heatmap[i][j] = (pred[0][0] + pred[1][0])/2
        np.save("/home/sonic/Coding/Decomposition/Simul_Data/heatmap.npy", heatmap)