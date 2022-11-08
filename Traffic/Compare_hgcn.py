import numpy as np
import matplotlib.pyplot as plt
from model_utils import math_utils
from model_utils import data_utils
from model_utils import layers
from model_utils import global_params as gp
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--unload_model', default=False, action='store_false', help='Load model or not.')

parser.add_argument('--pred_type', default='global', help='Choose "local" or "global".')

parser.add_argument('--epoch', default=500, help='Number of epochs to train.')

args = parser.parse_args()

model_load = bool(args.unload_model)
pred_type  =  str(args.pred_type)
epoch = int(args.epoch)

gp.__init()

train_weeks = 3
test_start = 1008 * train_weeks
n_his = 13

CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]

#global
if pred_type == 'global':
    n_node = 19
    train_X, train_Y = data_utils.data_gen(file_path='datasets/PemsD4_cloud.csv', n_route=None, n_his=n_his, loop=True, n_days=7 * train_weeks)
    test_X, test_Y = data_utils.data_gen(file_path='datasets/PemsD4_cloud.csv', n_route=None, n_his=n_his, loop=True, n_days=7, offset=test_start)

    train_weight = data_utils.weight_gen_cloud(data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_days=21)[0])

#local
if pred_type == 'local':
    routes = data_utils.station_gen(key=CityList)
    n_node = len(routes)
    train_X, train_Y = data_utils.data_gen(file_path='datasets/PemsD4.csv', n_route=routes, n_his=n_his, loop=True, n_days=7 * train_weeks)
    test_X, test_Y = data_utils.data_gen(file_path='datasets/PemsD4.csv', n_route=routes, n_his=n_his, loop=True, n_days=7, offset=test_start)

    train_weight = data_utils.weight_gen(gp.AdjacentGraph, routes, 1) 

if pred_type == 'combine':

    Eroutes = data_utils.station_gen(key=CityList)
    n_node = 19 + len(Eroutes)

    train_X_E, train_Y_E = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=Eroutes, n_his=n_his, loop=True, n_days=7 * train_weeks, offset=0)
    test_X_E, test_Y_E = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=Eroutes, n_his=n_his, loop=True, n_days=7, offset=test_start)

    train_X_C, train_Y_C = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_his=n_his, loop=True, n_days=7 * train_weeks, offset=0)
    test_X_C, test_Y_C = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_his=n_his, loop=True, n_days=7, offset=test_start)

    train_X = np.concatenate((train_X_E, train_X_C),axis=3)
    train_Y = np.concatenate((train_Y_E, train_Y_C),axis=1)
    test_X = np.concatenate((test_X_E, test_X_C),axis=3)
    test_Y = np.concatenate((test_Y_E, test_Y_C),axis=1)

    train_weight_E = data_utils.weight_gen(gp.AdjacentGraph, Eroutes, 1) 
    train_weight_C = data_utils.weight_gen_cloud(data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_days=21)[0])

    train_weight_a = np.concatenate((train_weight_E,np.zeros((len(Eroutes),19))),axis=1)
    train_weight_b = np.concatenate((np.zeros((19,len(Eroutes))),train_weight_C),axis=1)

    train_weight = np.concatenate((train_weight_a,train_weight_b),axis=0)

#kernel update
gp.update(train_weight, None)

import torch
train_x = torch.from_numpy(train_X).float().cuda()
train_y = torch.from_numpy(train_Y).float().cuda()
test_x = torch.from_numpy(test_X).float().cuda()

from torch import nn
from torch.autograd import Variable

class HGCN(nn.Module):
    def __init__(self,x_shape:list, args:tuple, c_out:list, name='_hgcn_'):
        super(HGCN,self).__init__()
        Kt = args[0]
        Ks = args[1]
        K = args[2]
        self.layer1 = layers.HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer2 = layers.HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer3 = layers.FC_3(x_shape)
        self.name = name
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def save(self, epoch):
        state = {}
        state[epoch] = epoch
        state['hcgn'] = self.state_dict()
        torch.save(state, 'model_saves/model'+self.name+'.pkl')

    def load(self):
        state = torch.load('model_saves/model'+self.name+'.pkl')
        self.load_state_dict(state['hcgn'])

Input_Tensor = [1008,1,n_his,n_node]

model = HGCN(Input_Tensor,(3,1,3),[16,8,16],name='_hgcn_'+pred_type).cuda()
if model_load:
    model.load()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 开始训练
pbar = tqdm(range(epoch))
for e in pbar:
    for i in range(train_weeks):
        start_idx, end_idx = 1008*i, 1008*(i+1)
        var_x = Variable(train_x[start_idx:end_idx])
        var_y = Variable(train_y[start_idx:end_idx])
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description('Epoch: {}, Loss {:.5f}'.format(e + 1, loss.data))

    #-------------------------------------------
    var_data = Variable(test_x)

    pred_test = model(var_data).reshape(-1,n_node) # 测试集的预测结果

    pred_test = pred_test.cpu().data.numpy()
    static = math_utils.evaluation(test_Y,pred_test)

    with open('result.txt', 'a+') as f:
        for stat in static.keys():
            f.write(''+str(stat)+':'+str(static[stat])+'\n')
    #-------------------------------------------

#model = model.eval() # 转换成测试模式

model.save(epoch)