from model_utils import servers as sv
from model_utils import data_utils
from model_utils import global_params as gp
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

train_weeks = 8
test_start = 1008 * train_weeks
n_his = 13
CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]


gp.__init()

def local_pred():
    #import test instances
    routes_Fremont = data_utils.station_gen(key="City of Fremont")
    test_x, test_y = data_utils.data_gen('datasets/PemsD4.csv', n_route=routes_Fremont, loop=True, n_days=7, offset=test_start)
    test_x = torch.tensor(test_x)

    #Edge Server1
    Fremont = sv.EdgeServer(city="City of Fremont", n_weeks=train_weeks, n_his=n_his)
    if model_load:
        Fremont.load_model()

    for i in range(epoch):
        Fremont.train(epoch=1)
        #--------------------------------------------------------------
        _test_y = Fremont.eval(test_x)[0].detach().cpu().numpy()
        static = math_utils.evaluation(_test_y, test_y)

        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n')

    Fremont.save_model()

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
    train_data = torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=(train_weeks+1)*7, n_his=n_his)[0]).float().cuda()
    routes_Fremont = data_utils.station_gen(key="City of Fremont")
    routes_Oakland = data_utils.station_gen(key="City of Oakland")
    routes_Richmond = data_utils.station_gen(key="City of Richmond")
    routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
    routes_SanJose = data_utils.station_gen(key="City of San Jose")

    #concatenate Edge Servers' dataset
    earlyExits = []
    for idx in range(train_weeks+1):
        start, end = 144 * 7 * idx, 144 * 7 * (idx + 1)
        F_out = []
        F_out.append(Fremont.eval(train_data[start:end,:,:,routes_Fremont])[1])
        F_out.append(Oakland.eval(train_data[start:end,:,:,routes_Oakland])[1])
        F_out.append(Richmond.eval(train_data[start:end,:,:,routes_Richmond])[1])
        F_out.append(SanFrancisco.eval(train_data[start:end,:,:,routes_SanFrancisco])[1])
        F_out.append(SanJose.eval(train_data[start:end,:,:,routes_SanJose])[1])
        earlyExits.append(np.concatenate(F_out, axis=3))

    Fremont.destroy()
    Oakland.destroy()
    Richmond.destroy()
    SanFrancisco.destroy()
    SanJose.destroy()

    concatMatrix = np.concatenate(earlyExits)

    roadDict, _outputBayArea = data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_days=(train_weeks+1)*7)

    train_x = torch.tensor(concatMatrix[0:test_start]).float().cuda()
    train_y = torch.tensor(_outputBayArea[0:test_start]).float().cuda()
    test_x = torch.tensor(concatMatrix[-1008:]).float().cuda()
    test_y = torch.tensor(_outputBayArea[-1008:]).float().cpu().numpy()
    
    BayArea = sv.CloudServer(city=CityList, dataDict = roadDict, epoch=epoch, n_weeks=train_weeks, n_days=7)
    if model_load:
        BayArea.load_model()

    pbar = tqdm(range(epoch))

    for e in pbar:
        BayArea.train(train_x, train_y, epoch=1, pbar=False)

        _test_y = BayArea.eval(test_x)[0].detach().cpu().numpy()
        static = math_utils.evaluation(_test_y, test_y)

        pbar.set_description("Epoch: {}, Loss: {:.5f}".format(e+1, static['MSE']))

        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n')

    BayArea.save_model()

if args.pred_type == 'local':
    local_pred()
elif args.pred_type == 'global':
    global_pred()
else:
    print('pred_type == None')
