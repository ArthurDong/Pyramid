import numpy as np
from model_utils import math_utils
from model_utils import data_utils
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

train_weeks = 3
test_start = 1008 * train_weeks
n_his = 13

CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]

#global
if pred_type == 'global':
    n_node = 19
    train_X, train_Y = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_days=7 * train_weeks, n_his=n_his, loop=True, offset=0)
    test_X, test_Y = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_days=7, n_his=n_his, loop=True, offset=test_start)

#local
if pred_type == 'local':
    routes = data_utils.station_gen(key=CityList)
    n_node = len(routes)

    train_X, train_Y = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=routes, n_days=7 * train_weeks, n_his=n_his, loop=True, offset=0)
    test_X, test_Y = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=routes, n_days=7, n_his=n_his, loop=True, offset=test_start)

if pred_type == 'combine':

    Eroutes = data_utils.station_gen(key=CityList)
    n_node = 19 + len(Eroutes)

    train_X_E, train_Y_E = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=Eroutes, n_days=7 * train_weeks, n_his=n_his, loop=True, offset=0)
    test_X_E, test_Y_E = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=Eroutes, n_days=7, n_his=n_his, loop=True, offset=test_start)

    train_X_C, train_Y_C = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_days=7 * train_weeks, n_his=n_his, loop=True, offset=0)
    test_X_C, test_Y_C = data_utils.data_gen(file_path = 'datasets/PemsD4_cloud.csv', n_route=None, n_days=7, n_his=n_his, loop=True, offset=test_start)

    train_X = np.concatenate((train_X_E, train_X_C),axis=3)
    train_Y = np.concatenate((train_Y_E, train_Y_C),axis=1)
    test_X = np.concatenate((test_X_E, test_X_C),axis=3)
    test_Y = np.concatenate((test_Y_E, test_Y_C),axis=1)

import torch

train_X = np.transpose(np.squeeze(train_X), [0,2,1])
train_Y = np.expand_dims(train_Y, 2)
test_X = np.transpose(np.squeeze(test_X), [0,2,1])
test_Y = np.expand_dims(test_Y, 2)

train_x = torch.from_numpy(train_X).float().cuda()
train_y = torch.from_numpy(train_Y).float().cuda()
test_x = torch.from_numpy(test_X).float().cuda()

from torch import nn
from torch.autograd import Variable

var_x = Variable(train_x)
var_y = Variable(train_y)
var_data = Variable(test_x)

class gru(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2, name='_gru_'):
        super(gru,self).__init__()
        self.name = name
        self.layer1 = nn.GRU(input_size,hidden_size,num_layer)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

    def load(self):
        state = torch.load('model_saves/model'+self.name+'.pkl')
        self.load_state_dict(state['gru'])

    def save(self, epoch):
        state = {}
        state[epoch] = epoch
        state['gru'] = self.state_dict()
        torch.save(state, 'model_saves/model'+self.name+'.pkl')

model = gru(input_size=n_his, hidden_size=16, output_size=1, num_layer=7, name='_lstm_'+pred_type).cuda()

if model_load:
    model.load()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 开始训练
pbar = tqdm(range(epoch))
for e in pbar:
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    pbar.set_description('Epoch: {}, Loss {:.5f}'.format(e + 1, loss.data))

    #------------------------------------------------------------
    pred_test = model(var_data)
    pred_test = pred_test.cpu().detach().numpy()
    static = math_utils.evaluation(test_Y,pred_test)

    with open('result.txt', 'a+') as f:
        for stat in static.keys():
            f.write(''+str(stat)+':'+str(static[stat])+'\n')
    #------------------------------------------------------------

model.save(epoch)