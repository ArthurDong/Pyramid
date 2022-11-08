import torch
import numpy as np
from model_utils import global_params as gp
from model_utils import servers as sv
from model_utils import data_utils
from model_utils import math_utils
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--unload_model', default=False, action='store_false', help='Load model or not.')
parser.add_argument('--epoch', default=500, help='Number of epochs to train.')
args = parser.parse_args()

model_load = bool(args.unload_model)
epoch = int(args.epoch)

gp.__init()

#train_param
n_weeks = 8
n_days = 7
n_his = 13
CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]

#Edge Server1
Fremont = sv.Server(city="City of Fremont", epoch=epoch, n_weeks=n_weeks, n_his=n_his)
Fremont.load_model()

#Edge Server2
Oakland = sv.Server(city="City of Oakland", epoch=epoch, n_weeks=n_weeks, n_his=n_his)
Oakland.load_model()

#Edge Server3
Richmond = sv.Server(city="City of Richmond", epoch=epoch, n_weeks=n_weeks, n_his=n_his)
Richmond.load_model()

#Edge Server4
SanFrancisco = sv.Server(city="City of San Francisco", epoch=epoch, n_weeks=n_weeks, n_his=n_his)
SanFrancisco.load_model()

#Edge Server5
SanJose = sv.Server(city="City of San Jose",epoch=epoch, n_weeks=n_weeks, n_his=n_his)
SanJose.load_model()

#prepare Cloud Servers' dataset
train_data = torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=n_days*n_weeks, n_his=n_his)[0])
routes_Fremont = data_utils.station_gen(key="City of Fremont")
routes_Oakland = data_utils.station_gen(key="City of Oakland")
routes_Richmond = data_utils.station_gen(key="City of Richmond")
routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
routes_SanJose = data_utils.station_gen(key="City of San Jose")

#concatenate Edge Servers' dataset
earlyExits = []
for idx in range(n_weeks):
    start, end = 144 * n_days * idx, 144 * n_days * (idx + 1)
    F_out = []
    F_out.append(Fremont.eval(train_data[start:end,:,:,routes_Fremont])[1])
    F_out.append(Oakland.eval(train_data[start:end,:,:,routes_Oakland])[1])
    F_out.append(Richmond.eval(train_data[start:end,:,:,routes_Richmond])[1])
    F_out.append(SanFrancisco.eval(train_data[start:end,:,:,routes_SanFrancisco])[1])
    F_out.append(SanJose.eval(train_data[start:end,:,:,routes_SanJose])[1])
    earlyExits.append(torch.cat(F_out, axis=3).detach().cpu().numpy())

Fremont.destroy()
Oakland.destroy()
Richmond.destroy()
SanFrancisco.destroy()
SanJose.destroy()

#Cloud_Server
roadDict, test_Y = data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_days=n_weeks*n_days)

#prepare_dataset
train_X = torch.tensor(np.concatenate(earlyExits)).float().cuda()
test_Y = torch.tensor(test_Y).float().cuda()

BayArea = sv.CloudServer(city=CityList, dataDict = roadDict, epoch=epoch, n_weeks=n_weeks, n_days=n_days, n_his=n_his)
if model_load:
    BayArea.load_model()

#train
BayArea.train(train_X, test_Y)
BayArea.save_model()

'''
#prepare Cloud Servers' dataset
train_data = torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=n_days)[0])
routes_Fremont = data_utils.station_gen(key="City of Fremont")
routes_Oakland = data_utils.station_gen(key="City of Oakland")
routes_Richmond = data_utils.station_gen(key="City of Richmond")
routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
routes_SanJose = data_utils.station_gen(key="City of San Jose")

#Edge pred
outputList = []
outputList.append(Fremont.eval(train_data[:,:,:,routes_Fremont])[1])
outputList.append(Oakland.eval(train_data[:,:,:,routes_Oakland])[1])
outputList.append(Richmond.eval(train_data[:,:,:,routes_Richmond])[1])
outputList.append(SanFrancisco.eval(train_data[:,:,:,routes_SanFrancisco])[1])
outputList.append(SanJose.eval(train_data[:,:,:,routes_SanJose])[1])

test_X = torch.cat(outputList, axis=3)

#Cloud pred
_test_Y, _ = BayArea.eval(test_X)

#Orignal data
_, test_Y = data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_weeks=n_weeks*n_days)

#evaluate routes
print("----Cloud Server of Bay Area----")
math_utils.result_print(math_utils.evaluation(test_Y, _test_Y))'''