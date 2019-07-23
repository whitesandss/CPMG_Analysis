import numpy as np

A_search_min = -300000
A_search_max = 300000          # Set the boundary between -100000 ~ 100000 (Hz) for A
B_search_min = 0
B_search_max = 200000           # Set the boundary between 0 ~ 200000 (Hz) for B
A_steps = 1000
B_steps = 1000

n_A_samples = (A_search_max - A_search_min) // A_steps
n_B_samples = (B_search_max - B_search_min) // B_steps

TOTAL_TEST_ARRAY = np.zeros((n_A_samples * n_B_samples, 2))

B_FIXED_VALUE = B_search_min
for i in range(n_B_samples):
    test_array = np.array([np.arange(A_search_min, A_search_max, A_steps), np.full((n_A_samples,), B_FIXED_VALUE)])
    test_array = test_array.transpose()
    test_array = np.round(test_array,2)
    TOTAL_TEST_ARRAY[i*n_A_samples:(i+1)*n_A_samples] = test_array
    B_FIXED_VALUE += B_steps
TOTAL_TEST_ARRAY.shape