import numpy as np
import torch
import model_utils.data_utils as data_utils

def __init(mode = 'tmp'):
    import pandas as pd
    global AdjacentGraph
    if mode == 'auto':
        AdjacentGraph = data_utils.get_weight_metrix('datasets/tmpBeijing', threshold=1000, index = 'id', longitude_col="Longitude", latitude_col='Latitude')
        np.savetxt('AJG_BJ',AdjacentGraph, delimiter=',')
    elif mode == 'tmp':
        AdjacentGraph = np.array(pd.read_csv('datasets/AJG_BJ', index_col=None, header=None))

def update(kernel:np.ndarray, metrix:np.ndarray):
    if kernel is not None:
        global Kernel 
        Kernel = torch.tensor(kernel).float().cuda()
    if metrix is not None:
        global Metrix
        Metrix = torch.tensor(metrix).float().cuda()

def set_max_value(i):
    global max_value
    max_value = i