from os import replace
import numpy as np
import pandas as pd
from tqdm import tqdm

def data_gen(file_path:str,  n_his:int, n_days:int, day_slot:int=24, offset:int=0,
             loop:bool=False, n_row=None, index_col=None, header=None, type='Aqi') -> np.ndarray:
    try:
        df = pd.read_csv(file_path, index_col=index_col, header=header).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    
    if isinstance(n_row, np.ndarray):
        df = df[:,n_row]
    elif isinstance(n_row, int):
        df = df[:,n_row:]

    if type=='Aqi':
        max_value = df.max()
        if max_value > 1:
            df /= max_value
            import model_utils.global_params as gp
            gp.set_max_value(max_value)
    elif type=='Weather':
        df = pd.DataFrame(df)
        df = df.replace({'Sunny/clear':0, 'Haze':1, 'Snow':2, 'Fog':3, 'Rain':4, 'Dust':5, 'Sand':6, 'Sleet':7, 'Rain/Snow with Hail':8, 'Rain with Hail':9})
        df = df.values

    train=[]
    for i in range(n_his):
        start, end = offset + i, offset + n_days * day_slot + i
        train.append(df[start:end])
    
    train = np.stack(train,axis=1)
    train = np.expand_dims(train, axis=1)

    start, end = offset + n_his, offset + n_days * day_slot + n_his
    test = np.array(df[start:end])


    return train, test

def get_weight_metrix(file_path:str = 'datasets/BeijingAq_S', threshold=10000, index = "station_id", longitude_col="Longitude", latitude_col='Latitude'):
    df = pd.read_csv(file_path, index_col=index)
    df = df[[longitude_col,latitude_col]]
    data = []
    print('prepare AdjacentMaxtrix')
    for i in tqdm(df.index):
        dataSeries=[]
        locA = (df.loc[i][longitude_col], df.loc[i][latitude_col])
        for j in df.index:
            locB = (df.loc[j][longitude_col], df.loc[j][latitude_col])
            dataSeries.append([1 if __distance(locA, locB)<threshold else 0])
        data.append(np.array(dataSeries))
    return np.concatenate(data,axis=1)

def __distance(Loc_a:tuple, Loc_b:tuple) -> float:
    '''
    lon = longitude
    lat = latitude  
    '''
    import math
    lng_a, lat_a, lng_b, lat_b = map(math.radians, [Loc_a[0], Loc_a[1], Loc_b[0], Loc_b[1]])
    
    R = 6378137.0
    a = math.sin(((lat_a-lat_b)/2))
    b = math.sin(((lng_a-lng_b)/2))
    dis = 2*R*math.asin(math.sqrt(math.pow(a,2)+math.cos(lat_a)*math.cos(lat_b)*math.pow(b,2)))
    return round(dis,3)

def weight_gen(AdjacentGraph:np.ndarray, n_route:np.ndarray=None, approximation=1, threshold=2000) -> np.ndarray:
    """
    approximation: int (1 First-order, 2+=Ks Chebyshev)
    """
    
    if isinstance(n_route, np.ndarray):
        W = AdjacentGraph[n_route][:,n_route]
    else:
        W = AdjacentGraph

    W_mask = np.ones([W.shape[0], W.shape[0]]) - np.identity(W.shape[0])  
    W = W_mask * (W < threshold)

    if approximation > 1:
        W = __Chebyshev(W, approximation)
    elif approximation == 1:
        W = __First_order(W)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{approximation}".')
    return W

def __First_order(A:np.ndarray) -> np.ndarray:
    '''
    First-order approximation n function.
    input: A:np.ndarray shape[n_route, n_route]
    return: np.ndarray shape[n_route, n_route]
    '''
    #~A = I + A
    A = A + np.identity(A.shape[0])
    d = np.sum(A, axis=1)
    D = np.diag(d**(-0.5)) 
    return np.matmul(np.matmul(D,A),D) 

def __Chebyshev(A:np.ndarray, kernel_size:int) -> np.ndarray:
    '''
    Chebyshev polynomials approximation function.
    input: A:np.ndarray shape[n_route, n_route], kernel_size:int
    return: np.ndarray shape[n_route, kernel_size*n_route]
    '''
    n, d = A.shape[0], np.sum(A, axis=1)
    #city2Indexaled_laplacian
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    
    lambda_max = np.real(np.max(np.linalg.eigvals(L)))
    L = (2 / lambda_max)*L - np.identity(n)
    #chebyshev approximation
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    L_list = [np.copy(L0), np.copy(L1)]
    for i in range(kernel_size - 2):
        Ln = np.mat(2 * L * L1 - L0)
        L_list.append(np.copy(Ln))
        L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
    return np.concatenate(L_list, axis=-1)

def station_gen_Aq(file_path:str = "datasets/BeijingAq_S.csv", key:str = "D1") -> np.ndarray:
    """
    input: file_path: str, key:str.
    return: df: ndarray. 
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    if isinstance(key, list) :
        key_list = key
        df = []
        for key in key_list:
            df.append(station_gen_Aq(file_path, key))
        return np.concatenate(df)

    df = df.loc[df['District']==key].index
    return np.array(df)

""" def weight_gen_cloud(roadDict):
    bframe = []
    for roadList in roadDict:
        Cities = roadList.split(' / ')
        for i in range(len(roadDict[roadList])):
            frame = np.ndarray((0))

            for roadList2 in roadDict:
                Cities2 = roadList2.split(' / ')

                if Cities == Cities2:
                    opt = np.ones(len(roadDict[roadList]))
                    frame = np.concatenate([frame,opt],0) 
                elif Cities2[0] or Cities2[1] in Cities:
                    opt = np.ones(len(roadDict[roadList2]))
                    frame = np.concatenate([frame,opt],0)
                else:
                    opt = np.zeros(len(roadDict[roadList2]))
                    frame = np.concatenate([frame,opt],0)
            bframe.append(frame)
    return np.stack(bframe,1)
"""
def weight_gen_cloud_Aq():
    """
    D1---D3
    |\  /|
    | D4 |
    |/  \|
    D2---D5
    """
    D1 = np.array([[1,1,1,1,0]])
    D2 = np.array([[1,1,0,1,1]])
    D3 = np.array([[1,0,1,1,1]])
    D4 = np.array([[1,1,1,1,1]])
    D5 = np.array([[0,1,1,1,1]])
    return np.concatenate([D1,D2,D3,D4,D5])

def transformMetric_gen_Aq(file_path = "datasets/BeijingAq_S.csv", regionList = ['D1','D2','D3','D4','D5']):

    try:
        data = pd.read_csv(file_path)
    except Exception:
        print(f'ERROR: input {file_path} was not found.')
    
    transformMetric = []
    totalSize = data.shape[0]
    stackSize = 0
    for region in regionList:
        dataSize = data[data['District']==region].shape[0]

        tempMetric = np.zeros(stackSize)
        tempMetric = np.append(tempMetric,np.ones(dataSize))
        tempMetric = np.append(tempMetric,np.zeros(totalSize-dataSize-stackSize))
        
        stackSize += dataSize
        transformMetric.append(np.expand_dims(tempMetric, axis=1))
    return np.concatenate(transformMetric, axis=1)

if __name__ == "__main__":
    weight_gen_cloud_Aq()
    #transformMetric_gen_Aq()
    #outes = station_gen_Aq()
    #x, y = data_gen('datasets/BeijingAq.csv', routes, n_his=21, n_days=21, day_slot=24, loop=True, index_col='utc_time', header=0)
    
    #print(x.shape,y.shape)
    #W = weight_gen('PemsD4_W.csv', routes, 1, sigma2=100000)
    #print(W.shape) i
    #file_name = {"station":"PemsD4_S.csv", "weight":"PemsD4_W.csv", "data":"PemsD4.csv"}
    #df = pd.read_csv("datasets/PemsD4_W.csv", header=None).values
    #print(data_combine(key="City of San Francisco", key2="City of Oakland", thresholds=1)[1].shape)
    #print(data_combine(key="City of Richmond", key2="City of Oakland")[2])
    #print(data_combine(key="City of Oakland", key2="City of Fremont")[2])    
    #print(data_combine(key="City of Fremont", key2="City of San Jose")[2])
    #AdjacentGraph = np.array(pd.read_csv('datasets/PemsD4_W_temp.csv',index_col=None, header=None))
    #print(highway_data_generator(AdjacentGraph, cityList = ["City of Fremont", "City of Oakland","City of Richmond", "City of San Francisco", "City of San Jose"])[0])
    #roadDict = {'City of Fremont / City of Oakland': ['I880-N', 'I880-S'], 'City of Fremont / City of San Jose': ['I880-N', 'I880-S', 'I680-S'], 'City of Oakland / City of Richmond': ['I580-E', 'I580-W', 'I80-W', 'I80-E'], 'City of Oakland / City of San Francisco': ['I80-W', 'I80-E'], 'City of Oakland / City of San Jose': ['I880-S', 'I880-N'], 'City of Richmond / City of San Francisco': ['I80-E', 'I80-W'], 'City of San Francisco / City of San Jose': ['US101-S', 'I280-N', 'I280-S', 'US101-N']}
    #id  = list(roadDict.keys())[0]
    #print(id, roadDict[id][0])
    #print(transformMetric_gen(roadDict, cityList = ["City of Fremont", "City of Oakland","City of Richmond", "City of San Francisco", "City of San Jose"]).shape)