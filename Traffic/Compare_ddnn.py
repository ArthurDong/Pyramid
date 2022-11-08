from model_utils import servers as sv
from model_utils import global_params as gp
from model_utils import data_utils
from model_utils import math_utils
from tqdm import tqdm
import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--unload_model', default=False, action='store_false', help='Load model or not.')

parser.add_argument('--pred_type', default='global', help='Choose "local" or "global".')

parser.add_argument('--epoch', default=500, help='Number of epochs to train.')

args = parser.parse_args()

model_load = bool(args.unload_model)
pred_type  =  str(args.pred_type)
epoch = int(args.epoch)
n_his = 13

#train
train_weeks = 3
test_start = 1008 * train_weeks

gp.__init()

def local_pred():
#Edge Server1
    City = "City of Oakland"
    routes_city = data_utils.station_gen(key=City)
    test_X, test_Y = data_utils.data_gen('datasets/PemsD4.csv', n_route=routes_city, loop=True, n_days=7, n_his=n_his, offset=test_start)
    test_X = torch.tensor(test_X)

    Edge_server = sv.EdgeServerDDNN(name="fremont",city=City, n_weeks=train_weeks, n_his=n_his)
    if model_load:
        Edge_server.load_model()
    for i in tqdm(range(epoch)):
        Edge_server.train(epoch=1, pbar=False)

        _test_data = Edge_server.eval(test_X)[0]
        _test_data = _test_data.detach().cpu().numpy()
            
        static = math_utils.evaluation(test_Y,_test_data)
        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n')

    Edge_server.save_model()

#global
def global_pred():
    
    #Edge Server1
    Fremont = sv.Server(city="City of Fremont", epoch=epoch, n_weeks=train_weeks+1, n_his=n_his)
    Fremont.load_model()

    #Edge Server2
    Oakland = sv.Server(city="City of Oakland", epoch=epoch, n_weeks=train_weeks+1, n_his=n_his)
    Oakland.load_model()

    #Edge Server3
    Richmond = sv.Server(city="City of Richmond", epoch=epoch, n_weeks=train_weeks+1, n_his=n_his)
    Richmond.load_model()

    #Edge Server4
    SanFrancisco = sv.Server(city="City of San Francisco", epoch=epoch, n_weeks=train_weeks+1, n_his=n_his)
    SanFrancisco.load_model()

    #Edge Server5
    SanJose = sv.Server(city="City of San Jose",epoch=epoch, n_weeks=train_weeks+1, n_his=n_his)
    SanJose.load_model()

    #prepare Cloud Servers' dataset
    train_data= torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=7 * (train_weeks+1), n_his=n_his)[0])
    routes_Fremont = data_utils.station_gen(key="City of Fremont")
    routes_Oakland = data_utils.station_gen(key="City of Oakland")
    routes_Richmond = data_utils.station_gen(key="City of Richmond")
    routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
    routes_SanJose = data_utils.station_gen(key="City of San Jose")

    #concatenate Edge Servers' dataset
    earlyExits = []
    for idx in range(train_weeks):
        start, end = 144 * 7 * idx, 144 * 7 * (idx + 1)
        F_out = []
        F_out.append(Fremont.eval(train_data[start:end,:,:,routes_Fremont])[1].detach().cpu())
        F_out.append(Oakland.eval(train_data[start:end,:,:,routes_Oakland])[1].detach().cpu())
        F_out.append(Richmond.eval(train_data[start:end,:,:,routes_Richmond])[1].detach().cpu())
        F_out.append(SanFrancisco.eval(train_data[start:end,:,:,routes_SanFrancisco])[1].detach().cpu())
        F_out.append(SanJose.eval(train_data[start:end,:,:,routes_SanJose])[1].detach().cpu())
        earlyExits.append(np.concatenate(F_out, axis=3))

    concatMatrix = np.concatenate(earlyExits)

    train_X = torch.tensor(concatMatrix[:test_start]).float().cuda()
    test_X = torch.tensor(concatMatrix[test_start:]).float().cuda()

    Fremont.destroy()
    Oakland.destroy()
    Richmond.destroy()
    SanFrancisco.destroy()
    SanJose.destroy()

    routes_Bayarea = data_utils.station_gen(key=["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"])
    _, test_data = data_utils.data_gen('datasets/PemsD4.csv', n_route=routes_Bayarea, loop=True, n_days=7, n_his=n_his, offset=test_start)
    #Cloud Server
    Bayarea = sv.CloudServerDDNN(name="model_ddnn_bayarea" ,
                                city=["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"],
                                n_weeks=train_weeks, n_his = n_his-6)
    if model_load:
        Bayarea.load_model()

    pbar = tqdm(range(epoch))
    
    for e in pbar:
        Bayarea.train(train_X, epoch=1, pbar=False)

        """ _test_data = Bayarea.eval(test_data_X)[0].detach().cpu().numpy()
        static = math_utils.evaluation(test_data, _test_data)

        pbar.set_description("Epoch: {}, Loss: {:.5f}".format(e+1, static['MSE']))
        
        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n') """

    Bayarea.save_model()

def combine_pred():
    #Edge Servers and data
    ES_route = []
    ES_route.append(data_utils.station_gen(key="City of Fremont"))
    ES_route.append(data_utils.station_gen(key="City of Oakland"))
    ES_route.append(data_utils.station_gen(key="City of Richmond"))
    ES_route.append(data_utils.station_gen(key="City of San Francisco"))
    ES_route.append(data_utils.station_gen(key="City of San Jose"))

    ES_list = []
    ES_list.append(sv.EdgeServerDDNN(name="fremont",city="City of Fremont", n_weeks=train_weeks))
    ES_list.append(sv.EdgeServerDDNN(name="fremont",city="City of Oakland", n_weeks=train_weeks))
    ES_list.append(sv.EdgeServerDDNN(name="fremont",city="City of Richmond", n_weeks=train_weeks))
    ES_list.append(sv.EdgeServerDDNN(name="fremont",city="City of San Francisco", n_weeks=train_weeks))
    ES_list.append(sv.EdgeServerDDNN(name="fremont",city="City of San Jose", n_weeks=train_weeks))

    data_X, data_Y = data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=7 * (train_weeks+1))

    train_X = torch.tensor(data_X[:test_start]).float().cuda()
    test_X = torch.tensor(data_X[test_start:]).float().cuda()
    test_Y = data_Y[test_start:]

    #Cloud Servers and data
    routes_Bayarea = data_utils.station_gen(key=["City of Fremont",
                                                "City of Oakland",
                                                "City of Richmond",
                                                "City of San Francisco",
                                                "City of San Jose"])
                                                
    test_data_cloud = test_Y[:,routes_Bayarea] #data_utils.data_gen('datasets/PemsD4.csv', n_route=routes_Bayarea,
                                            #loop=True, n_days=7, offset=test_start)[1]
    #Cloud Server
    Bayarea = sv.CloudServerDDNN(name="BayArea" ,city=["City of Fremont",
                                                        "City of Oakland",
                                                        "City of Richmond",
                                                        "City of San Francisco",
                                                        "City of San Jose"],
                                                        n_weeks=train_weeks)

    if model_load:
        for ES in ES_list:
            ES.load_model()
        Bayarea.load_model()

    
    for e in tqdm(range(epoch)):
        torch.cuda.empty_cache()
        #train Edge Servers
        train_X_cloud_list = []
        test_X_cloud_list = []
        for i, ES in enumerate(ES_list):
            ES.train(epoch=1, pbar=False)

            train_X_cloud_list.append(ES.eval(train_X[:,:,:,ES_route[i]])[1])

            _test_data, test_X_cloud_data = ES.eval(test_X[:,:,:,ES_route[i]])
            
            #_test_data = _test_data.detach().cpu().numpy()
            """ static = math_utils.evaluation(test_Y[:,ES_route[i]],_test_data)
            with open('result.txt', 'a+') as f:
                for stat in static.keys():
                    f.write(''+str(stat)+':'+str(static[stat])+'\n') """

            test_X_cloud_list.append(test_X_cloud_data)

        #train Cloud Server
        train_X_cloud = torch.cat(train_X_cloud_list,dim=3).detach()
        test_X_cloud = torch.cat(test_X_cloud_list,dim=3).detach()
        Bayarea.train(train_X_cloud, epoch=1, pbar=False)

        _test_data_cloud = Bayarea.eval(test_X_cloud)[0].detach().cpu().numpy()
        static = math_utils.evaluation(_test_data_cloud, test_data_cloud)

        #pbar.set_description("Epoch: {}, Loss: {:.5f}".format(e+1, static['MSE']))
        
        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n')



    for ES in ES_list:
        ES.save_model()

if pred_type=='local':
    local_pred()

if pred_type=='global':
    global_pred()

if pred_type=='combine':
    combine_pred()
