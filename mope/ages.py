from __future__ import unicode_literals
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np

ages = np.array((
    0.4, 9, 14, 15, 17, 22, 23, 24, 24, 25, 25, 26, 26, 28, 30, 30, 30, 32, 32,
    33, 34, 35, 37, 38, 40, 40, 41, 42, 42, 43, 43, 43, 44, 45, 45, 46, 46, 47,
    47, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52,
    52, 52, 53, 53, 53, 55, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 59, 60, 60,
    60, 60, 61, 61, 62, 62, 62, 62, 63, 64, 64, 64, 64, 64, 65, 66, 66, 66, 67,
    67, 68, 69, 69, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 74, 74, 74,
    75, 76, 76, 76, 77, 77, 77, 77, 79, 81, 81, 81, 82, 82, 83, 83, 83, 84, 85,
    86, 87, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 92, 92, 93, 96
    ), dtype = np.float64)
