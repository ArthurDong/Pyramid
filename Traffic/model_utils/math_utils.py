import numpy as np
import pandas as pd

def MSE(_y:np.ndarray, y:np.ndarray) -> float:
    #Mean Square Error
    return  np.mean((_y - y) ** 2)

def RMSE(_y:np.ndarray, y:np.ndarray) -> float:
    #Root Mean Square Error
    return  np.sqrt(np.mean((_y - y) ** 2))

def MAE(_y:np.ndarray, y:np.ndarray) -> float:
    #Mean Absolute Error
    return  np.mean(np.abs(_y - y))

def MAPE(_y:np.ndarray, y:np.ndarray) -> float:
    #Mean Absolute Percentage Error
    return  np.mean(np.abs((_y - y) / y))

def R_Squared(_y:np.ndarray, y:np.ndarray) -> float:
    #R_Squared
    y_var = MSE(y, np.ones_like(y) * np.mean(y))
    return  1 - MSE(_y, y) / y_var

def evaluation(_y:np.ndarray, y:np.ndarray) -> dict:
    #iteratively eval result 
    static = {}
    static['MSE'] = MSE(_y,y)
    static['RMSE'] = RMSE(_y,y)
    static['MAE'] = MAE(_y,y)
    static['MAPE'] = MAPE(_y,y)
    static['R_Squared'] = R_Squared(_y,y)
    for stat in static.keys():
        res = ''+str(stat)+':'+str(static[stat])
        #print(res)
    return static

def result_print(statics):
    for stat in statics.keys():
        print(str(stat)+':'+str(statics[stat]))