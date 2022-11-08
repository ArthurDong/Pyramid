import torch
import numpy as np
from model_utils import math_utils
from model_utils import data_utils
from model_utils import layers
from model_utils import global_params as gp
from tqdm import tqdm
from torch import nn

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

train_weeks = 30
test_start = 168 * train_weeks
n_his = 13

class ATTCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, K:int, c_out:int):
        super(ATTCov,self).__init__()
        self.K = K
        self.c_out = c_out

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        #layers
        self.conv_layer1 = nn.Conv2d(self.T_slot, 1, (K,1), stride=(1,1), bias=True)
        self.conv_layer2 = nn.Conv2d(self.c_in, 1, (K,1), stride=(1,1), bias=True)

        self.weight = nn.Parameter(torch.rand(self.c_in-2,self.T_slot-2), requires_grad=True)
        #self.bias=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        self.actv_layer = nn.Sigmoid()

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        conv_out_a = self.conv_layer1(x_input.permute(0,2,1,3)).squeeze(dim=1) # batch_size, c_in, n_node
        conv_out_b = self.conv_layer2(x_input.permute(0,1,2,3)).squeeze(dim=1) # batch_size, T_slot, n_node
        conv_out_a = conv_out_a.permute(0,2,1) # batch_size, n_node, c_in
        conv_out_b = conv_out_b.permute(0,2,1) # batch_size, n_node, T_slot
        x_output = self.actv_layer(torch.matmul(conv_out_a, self.weight) * conv_out_b) # batch_size, n_node, fs
        return x_output.permute(0,2,1).unsqueeze(1) # batch_size, 1,fs, n_node

class HGCN_block(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kt:int, Ks:int, K:int, c_out:list):
        super(HGCN_block,self).__init__()
        self.layer0 = layers.TemporalCov(x_shape , Kt, c_out[0])
        x_shape[2] -= Kt-1
        x_shape[1] = c_out[0]
        self.layer1 = layers.SpatioCov(x_shape , Ks, c_out[1])
        x_shape[1] = c_out[1]
        self.layer2 = ATTCov(x_shape , K, c_out[2])
        x_shape[2] -= Kt-1
        x_shape[1] = 1
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.layer0(x_input)
        x_input = self.layer1(x_input)
        x_output = self.layer2(x_input)
        return x_output

class FC_3(nn.Module):
    #x_input [batch_size, 1, fs, n_node]
    def __init__(self, x_shape:list, fs = [8,4,1]):
        super(FC_3,self).__init__()
        self.batch_size = x_shape[0]
        self.i_fs = x_shape[2]
        self.o_fs = fs[2]
        self.n_node =  x_shape[3]

        self.fc1 = nn.Linear(self.i_fs, fs[0])
        self.fc2 = nn.Linear(fs[0],fs[1])
        self.fc3 = nn.Linear(fs[1],fs[2])

        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()

    def forward(self, x_input:torch.Tensor):
        x_input = x_input.permute(0,3,1,2).squeeze() #x_input [batch_size, n_node, fs]
        x_input = x_input.reshape(-1,self.i_fs) #x_input [batch_size*n_node, fs]
        x_input = self.fc1(x_input) 
        x_input = self.actv1(x_input)
        x_input = self.fc2(x_input)
        x_input = self.actv2(x_input)
        x_input = self.fc3(x_input) #x_input [batch_size*n_node, 1]
        if self.o_fs == 1:
            return x_input.squeeze().reshape(self.batch_size,self.n_node) #x_input [batch_size, n_node]
        else:
            return x_input.reshape(self.batch_size, self.n_node, self.o_fs) #x_input [batch_size, n_node, class_num]

class HGCN(nn.Module):
    def __init__(self, x_shape:list, args:tuple, c_out:list, name='_hgcn_'):
        super(HGCN,self).__init__()
        Kt = args[0] 
        Ks = args[1]
        K = args[2]
        self.layer1 = HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer2 = HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer3 = FC_3(x_shape)
        self.name = name
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
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

class HGCN_CLS(nn.Module):
    def __init__(self,x_shape:list, args:tuple, c_out:list, name='_hgcn_'):
        super(HGCN_CLS,self).__init__()
        Kt = args[0] 
        Ks = args[1]
        K = args[2]
        self.layer1 = HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer2 = HGCN_block(x_shape, Kt, Ks, K, c_out)
        self.layer3 = FC_3(x_shape, fs=[16,8,10])
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

#global
if pred_type == 'global':
    train_X, train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7 * train_weeks, n_his=n_his, offset=0,
                                             index_col='utc_time', header=0)
    test_X, test_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7, n_his=n_his, offset=test_start,
                                             index_col='utc_time', header=0)

    train_weight = data_utils.weight_gen_cloud_Aq()
    criterion = nn.CrossEntropyLoss().cuda()
    #kernel update
    gp.update(train_weight, None)
    model = HGCN_CLS([168,1,n_his,5], (3,1,3), [16,8,16], name='_hgcn_'+pred_type).cuda()

#local
if pred_type == 'local':
    District = ['D1','D2','D3','D4','D5']
    routes = data_utils.station_gen_Aq(key=District)
    n_node = len(routes)

    train_X, train_Y = data_utils.data_gen(file_path = 'datasets/BeijingAq.csv', n_row=routes,
                                     n_days=7 * train_weeks, n_his=n_his, offset=0,
                                     header=0, index_col='utc_time')
    test_X, test_Y = data_utils.data_gen(file_path = 'datasets/BeijingAq.csv', n_row=routes, 
                                     n_days=7, n_his=n_his, offset=test_start, 
                                     header=0, index_col='utc_time')

    train_weight = data_utils.weight_gen(gp.AdjacentGraph, routes, 1) 
    criterion = nn.MSELoss().cuda()
    #kernel update
    gp.update(train_weight, None)
    model = HGCN([168,1,n_his,n_node], (3,1,3), [16,8,16], name='_hgcn_'+pred_type).cuda()


import torch
train_X = torch.from_numpy(train_X).float().cuda()
train_Y = torch.from_numpy(train_Y).float().cuda()
test_X = torch.from_numpy(test_X).float().cuda()

if model_load:
    model.load()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 开始训练
pbar = tqdm(range(epoch))
#pbar = range(epoch)
for e in pbar:
    for idx in range(train_weeks):
        start, end = 168 * idx, 168 * (idx+1)
        X = train_X[start:end]
        Y = train_Y[start:end]
        # 前向传播
        if pred_type == 'global':
            out = model.forward(X).view(-1, 10)

            label = Y.contiguous().view(-1).long()
            loss = criterion(out, label)
        else:
            out = model(X)
            loss = criterion(out, Y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description('Epoch: {}, Loss {:.5f}'.format(e + 1, loss.data))

        #model = model.eval() # 转换成测试模式
        if pred_type == 'local':
            pred_test = model.forward(test_X)
            pred_test = pred_test.cpu().detach().numpy()
            static = math_utils.evaluation(test_Y, pred_test, type='regression')
        if pred_type == 'global':
            pred_test = model.forward(test_X).view(-1, 10)
            pred_test = pred_test.cpu().detach().numpy()
            pred_test = np.argmax(pred_test, 1)
            test_Y = np.reshape(test_Y, (-1))
            static = math_utils.evaluation(test_Y, pred_test, type='classification')

        with open('result.txt','a+') as f:
            f.write('loss:'+str(loss.data)+'\n')
            for stat in static.keys():
                f.write(str(stat)+':'+str(static[stat])+'\n')

model.save(epoch)

#------------------------------------------------------------------------
#model = model.eval() # 转换成测试模式
if pred_type == 'local':
    pred_test = model(test_X)
    pred_test = pred_test.cpu().detach().numpy()
    static = math_utils.evaluation(test_Y, pred_test, type='regression')
if pred_type == 'global':
    pred_test = model(test_X).view(-1, 10)
    pred_test = pred_test.cpu().detach().numpy()
    pred_test = np.argmax(pred_test, 1)
    test_Y = np.reshape(test_Y, (-1))
    static = math_utils.evaluation(test_Y, pred_test, type='classification')

for stat in static.keys():
    print(str(stat)+':'+str(static[stat]))
