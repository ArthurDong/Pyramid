import numpy as np
import torch
import model_utils.data_utils as data_utils

def __init(mode=0):
    global AdjacentGraph
    import pandas as pd
    if mode==0:
        AdjacentGraph = np.array(pd.read_csv('datasets/tmpPems',index_col=None, header=None))
    else:
        AdjacentGraph = np.array(pd.read_csv('datasets/tmpBeijing', index_col=None, header=None))

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