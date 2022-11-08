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
    return  np.mean(np.abs(_y - y) / (y + 1e-5))

def R_Squared(_y:np.ndarray, y:np.ndarray) -> float:
    #R_Squared
    y_var = MSE(y, np.ones_like(y) * np.mean(y))
    return  1 - MSE(_y, y) / y_var

def Accuracy(_y:np.ndarray, y:np.ndarray) -> float:
    y_same = 0
    for i, value in enumerate(y):
        if value == _y[i]:
            y_same += 1
    return  y_same / len(y)

def evaluation(_y:np.ndarray, y:np.ndarray, type='regression') -> dict:
    #iteratively eval result 
    static = {}

    if type == 'regression':
        static['MSE'] = MSE(_y,y)
        static['RMSE'] = RMSE(_y,y)
        static['MAE'] = MAE(_y,y)
        static['MAPE'] = MAPE(_y,y)
        static['R_Squared'] = R_Squared(_y,y)
    elif type == 'classification':
        static['R_Squared'] = R_Squared(_y,y)
        static['Accuracy'] = Accuracy(_y,y)

        """ for stat in static.keys():
            res = ''+str(stat)+':'+str(static[stat])
            print(res) """
    return static

if __name__ == "__main__":
    #a = np.linspace(0,0.95,50)
    #b = np.linspace(0,1,50)
    #evaluation(a,b)
    a = [0,1,0,0,1]
    b = [0,1,0,1,1]
    print(Accuracy(a, b))