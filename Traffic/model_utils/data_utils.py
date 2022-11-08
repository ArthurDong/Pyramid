import numpy as np
import pandas as pd
from tqdm import tqdm

def data_gen(file_path:str = 'datasets/PemsD4.csv', n_route=None, n_his=21, n_days=3, day_slot=144, loop=False, index_col=None, header=None, offset=0) -> np.ndarray:
    try:
        df = pd.read_csv(file_path, index_col=index_col, header=header).values
        df = np.array(df)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    if isinstance(n_route, np.ndarray):
        df = df[:,n_route]
    
    max_value = df.max()
    if max_value > 1:
        df /= max_value
        import model_utils.global_params as gp
        gp.set_max_value(max_value)

    time_step = day_slot if loop else day_slot-n_his

    train = []
    test = []
    for d in range(n_days):
        ts_idx = (d+1)*day_slot - n_his + offset if loop else d * day_slot + offset
        #train_case
        for i in range(time_step):
            start, end = ts_idx + i, ts_idx + i + n_his
            train.append(df[start:end])
        #test_case
        start = ts_idx + n_his
        end = ts_idx + day_slot + n_his if loop else ts_idx + day_slot
        test.append(df[start:end])
    train = np.stack(train,axis=0)
    test = np.concatenate(test,axis=0)

    train = np.expand_dims(train, axis=1) 
    return train, test

def get_weight_metrix(file_path:str = 'datasets/PemsD4_S.csv', threshold=10000, index = "VDS",longitude_col="Longitude", latitude_col='Latitude'):
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

def station_gen(file_path:str = "datasets/PemsD4_S.csv", key:str = "City of San Jose") -> np.ndarray:
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
            df.append(station_gen(file_path, key))
        return np.concatenate(df)

    if 'City' in key:
        df = df.loc[df['City']==key].index
    elif 'County' in key:
        df = df.loc[df['County']==key].index
    elif key[:-2] in ('US101','I80','I280','I580','I680','I880','SR4','SR17','SR24','SR84','SR85','SR87','SR92','SR237','SR238'):
        df = df.loc[df['Roads']==key].index
    df = np.array(df)
    return df

def __roads_in_city(key:str = "City of Fremont", file_path:str = "datasets/PemsD4_S.csv") -> list:
    """
    input: file_path: str, key: str.
    return: df: list[str]. 
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    roads = df['Roads'].loc[(df.City==key)]
    roadList = []
    for road in roads:
        if road not in roadList:
            roadList.append(road)

    return roadList

def __common_road(key,key2) -> list:
    """
    input: key: str, key2: str.
    return: df: list[str]. 
    """
    roadList = []
    for i in __roads_in_city(key):
        if i in __roads_in_city(key2):
            roadList.append(i)
    return roadList

def __getdata(dataFrame, n_route, loop=True, n_days=7, n_his=21, day_slot=144) -> np.ndarray:
    """
    input: dataFrame: pd.DataFrame, n_route: list[int].
    return: test: np.ndarray. 
    """
    dataFrame = dataFrame[:,n_route]
    test = []
    if loop:
        start, end = day_slot, day_slot*(n_days+1)
        test = np.array(dataFrame[start:end])
    else:
        for d in range(n_days):
            off_set = d * day_slot
            start, end = off_set + n_his, off_set + day_slot
            test.append(dataFrame[start:end])
        test = np.concatenate(test,axis=0)
    return test

def highway_data_generator(AdjacentGraph:np.ndarray, cityList:list=["City of Fremont", "City of San Jose"], thresholds=3, loop=True, n_days=7, n_his=21, day_slot=144, **file_path):  
    #load data
    dataFrame_weight = pd.DataFrame(AdjacentGraph)

    try:
        dataFrame_data = np.array(pd.read_csv(file_path['data'], header=None).values)
    except Exception:
        file_path_D = "datasets/PemsD4.csv"
        #print(f'ERROR: input file was not found. use default value {file_path_D}.')
        dataFrame_data = np.array(pd.read_csv(file_path_D, header=None).values)

    try:
        dataFrame_Stats = pd.read_csv(file_path['station'])
    except Exception:
        file_path_S = "datasets/PemsD4_S.csv"
        #print(f'ERROR: input file was not found. use default value {file_path_S}.')
        dataFrame_Stats = pd.read_csv(file_path_S)
    #-----------------------------------------------------------------------------------
    cityStack = cityList.copy()

    transformedData = []
    roadDict = {}

    for city in cityList:
        cityStack.remove(city)
        for city2 in cityStack:
            if city2 is not None:
                common_road = __common_road(city, city2)
                #build road dictionary
                if common_road:
                    roadDict[str(city)+' / '+str(city2)] = common_road
                
                #build transformed data
                for road in common_road:
                    #find VDSIndex
                    city1Index = dataFrame_Stats.loc[(dataFrame_Stats['City']==city)  & (dataFrame_Stats['Roads']==road)].index
                    city2Index = dataFrame_Stats.loc[(dataFrame_Stats['City']==city2) & (dataFrame_Stats['Roads']==road)].index
                    #build index list
                    roadMIndex = __build_data(dataFrame_weight,city1Index,city2Index, thresholds=thresholds)
                    #build data matrix
                    transformedData.append(np.mean(__getdata(dataFrame_data, roadMIndex, loop=loop, n_days=n_days, n_his=n_his, day_slot=day_slot),axis=1))

    del cityStack
    transformedData = np.stack(transformedData,axis=-1)
        
    return roadDict, transformedData

def __build_data(dataFrame_W, city1Index, city2Index, thresholds=3):
    dataFrame = dataFrame_W.copy()
    #distance matrix between VDS of two Cities
    distanceMetric = np.array(dataFrame.iloc[city1Index, city2Index])
    #find closest VDS 
    idx = np.where(distanceMetric == np.min(distanceMetric))
    #roadMIndex = [city1Index[list_2_int(idx[0])], city2Index[list_2_int(idx[1])]]
    roadMIndex= []
    #distance list in city1Index
    city1Index = np.array(city1Index)[np.argsort(np.array(dataFrame.iloc[city1Index, city1Index[__list_2_int(idx[0])]]))]
    #distance list in city2Index
    city2Index = np.array(city2Index)[np.argsort(np.array(dataFrame.iloc[city2Index, city2Index[__list_2_int(idx[1])]]))]
    #add addtional VDS to RS 
    roadMIndex.extend(city1Index[:thresholds])
    roadMIndex.extend(city2Index[:thresholds])

    return roadMIndex

def __list_2_int(List:list):
    #unpack list
    if len(List) == 1:
        return List[0]
    else:
        print("Error!")

def weight_gen_cloud(roadDict):
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

def transformMetric_gen(roadDict, cityList = ["City of Fremont", "City of San Jose"], file_path = "datasets/PemsD4_S.csv"):

    try:
        dataFrame_Stats = pd.read_csv(file_path)
    except Exception:
        file_path_S = "datasets/PemsD4_S.csv"
        print(f'ERROR: input file was not found. use default value {file_path_S}.')
        dataFrame_Stats = pd.read_csv(file_path_S)
    
    cityDict = {}
    for city in cityList:
        cityDict[city] = dataFrame_Stats.loc[dataFrame_Stats['City']==city]

    transformMetric = []
    for road in roadDict:
        cities = str(road).split(" / ")
        city1, city2  = cities[0], cities[1]
        for highwayNumber in roadDict[road]:
            transformFrame = []
            for city in cityList:
                if city == city1:
                    indexStats = np.zeros(len(cityDict[city]))
                    indexStats[np.where(cityDict[city]['Roads']==highwayNumber)[0]] = 1
                elif city == city2:
                    indexStats = np.zeros(len(cityDict[city]))
                    indexStats[np.where(cityDict[city]['Roads']==highwayNumber)[0]] = 1
                else:
                    indexStats = np.zeros(len(cityDict[city]))
                transformFrame.extend(indexStats)
            transformMetric.append(np.array(transformFrame))
    transformMetric = np.stack(transformMetric).transpose()
    return transformMetric

if __name__ == "__main__":
    #weight_gen_cloud_Aq()
    #transformMetric_gen_Aq()
    #outes = station_gen_Aq()
    #x, y = data_gen('datasets/Beijing_dataset/BeijingAq.csv', routes, n_his=21, n_days=21, day_slot=24, loop=True, index_col='utc_time', header=0)
    
    #print(x.shape,y.shape)
    #W = weight_gen('PemsD4_W.csv', routes, 1, sigma2=100000)
    #print(W.shape) i
    #file_name = {"station":"PemsD4_S.csv", "weight":"PemsD4_W.csv", "data":"PemsD4.csv"}
    #print(data_combine(key="City of San Francisco", key2="City of Oakland", thresholds=1)[1].shape)
    #print(data_combine(key="City of Richmond", key2="City of Oakland")[2])
    #print(data_combine(key="City of Oakland", key2="City of Fremont")[2])    
    #print(data_combine(key="City of Fremont", key2="City of San Jose")[2])
    AdjacentGraph = np.array(pd.read_csv('datasets/tmpPems',index_col=None, header=None))
    #roadDict = {'City of Fremont / City of Oakland': ['I880-N', 'I880-S'], 'City of Fremont / City of San Jose': ['I880-N', 'I880-S', 'I680-S'], 'City of Oakland / City of Richmond': ['I580-E', 'I580-W', 'I80-W', 'I80-E'], 'City of Oakland / City of San Francisco': ['I80-W', 'I80-E'], 'City of Oakland / City of San Jose': ['I880-S', 'I880-N'], 'City of Richmond / City of San Francisco': ['I80-E', 'I80-W'], 'City of San Francisco / City of San Jose': ['US101-S', 'I280-N', 'I280-S', 'US101-N']}
    #id  = list(roadDict.keys())[0]
    #print(id, roadDict[id][0])
    #print(transformMetric_gen(roadDict, cityList = ["City of Fremont", "City of Oakland","City of Richmond", "City of San Francisco", "City of San Jose"]).shape)