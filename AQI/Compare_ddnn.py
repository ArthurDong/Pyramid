from model_utils import servers as sv
from model_utils import global_params as gp
from model_utils import data_utils
from model_utils import math_utils
from tqdm import tqdm
import os
import torch
import torch.nn as nn
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
train_weeks = 30
test_start = 168 * train_weeks

gp.__init()

class STDCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kernel_size:int, c_out:int, UseConv = False):
        super(STDCov,self).__init__()

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        #layers
        self.conv_layer = nn.Conv2d(self.c_in, c_out, (Kernel_size, 1), stride=(1,1), bias=True)
        self.linear_layer = nn.Linear(self.n_node, self.n_node, bias=True)
        self.actv_layer = nn.Sigmoid()
        if UseConv:
            self.bias_layer = nn.Conv2d(self.c_in, self.c_in, (1,1), stride=1, bias=False)
        else:
            self.bias_layer = None

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        if isinstance(self.bias_layer, nn.Conv2d):
            bias = self.bias_layer(x_input)
        else:
            bias = x_input

        #X -> [batch_size*c_in*T_slot, n_node] 
        reshape_out = torch.reshape(x_input,(-1, self.n_node))
        #SpatioFullConnection
        linear_out = self.linear_layer(reshape_out) 
        #X -> [batch_size, T_slot, c_in, n_node] -> [batch_size, c_in, T_slot, n_node]
        reshape_out = torch.reshape(linear_out,(-1, self.c_in, self.T_slot, self.n_node)) 

        #TemporalConnection
        conv_out = self.conv_layer(reshape_out + bias)

        return self.actv_layer(conv_out)

class OutputLayer(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list):
        super(OutputLayer,self).__init__()
        self.fc = nn.Conv2d(x_shape[1], 1, kernel_size=[x_shape[2],1], stride=[1,1])
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.fc(x_input).squeeze()
        logits = self.Sigmoid(x_input)
        return logits

class OutputLayerClass(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, cls):
        super(OutputLayerClass,self).__init__()
        self.fc = nn.Conv2d(x_shape[1], cls, kernel_size=[x_shape[2],1], stride=[1,1])
        self.classes = cls
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.fc(x_input).squeeze()
        logits = self.Sigmoid(x_input.permute(0,2,1).reshape((-1, self.classes)))
        return logits

class DCNN(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kernel_size:int, c_out:list, cls=1):
        super(DCNN,self).__init__()
        layers = []
        for i, out in enumerate(c_out):
            layer = STDCov(x_shape, Kernel_size, out)
            x_shape[2] = layer.T_slot - Kernel_size + 1
            x_shape[1] = out           
            layers.append(layer)
        
        self.layers = nn.Sequential(*layers)
        if cls == 1:
            self.output = OutputLayer(x_shape)
        else:
            self.output = OutputLayerClass(x_shape, cls)

    def forward(self, x_input:torch.Tensor):
        x_input = self.layers(x_input)
        x_output = self.output(x_input)
        return x_output, x_input


class EdgeServerDDNN:
    def __init__(self, name="ddnn_local", region="D1", epoch=500, c_out=[16,8,16,8,16], Kernel_size=3, n_days=7, n_weeks=3, n_his=21):
        #local param
        stations = data_utils.station_gen_Aq(key=region)
        Tensor_shape = [168 * n_days, 1, n_his, stations.size]
        #class param
        self.name = name
        self.epoch = epoch
        self.n_days = n_days
        self.n_weeks = n_weeks
        self.n_row = stations
        #kernel update
        train_weight = data_utils.weight_gen(gp.AdjacentGraph, stations)
        gp.update(train_weight, None)
        #model nn
        self.nn = DCNN(Tensor_shape, Kernel_size, c_out).cuda()
        #train, test data
        self.train_X, self.train_Y = data_utils.data_gen('datasets/BeijingAq.csv', n_row = self.n_row, n_days = 7 * train_weeks,
                                                        n_his=n_his, loop=True, offset=0,
                                                        header=0, index_col='utc_time')

        self.train_X = torch.from_numpy(self.train_X).float().cuda()
        self.train_Y = torch.from_numpy(self.train_Y).float().cuda()

    def load_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._load({'nn':self.nn}, file_path)

    def save_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._save({'nn':self.nn}, file_path)

    def train(self, train_X:torch.Tensor=None, train_Y:torch.Tensor=None, epoch=None, pbar=True):
        self.nn.train()
        if epoch == None:
            pass
        else:
            self.epoch = epoch
        self.pbar = pbar
        #training
        if train_X is not None:
            self.train_X = train_X
        if train_Y is not None:
            self.train_Y = train_Y

        self._training()
    
    def eval(self, test_X:torch.Tensor):
        self.nn.eval()
        test_X = test_X.float().cuda()
        
        return self.nn(test_X)

    def _training(self):
        #Optim & Lossfuc
        gd = torch.optim.Adam(self.nn.parameters())
        loss_fn = torch.nn.MSELoss().cuda()
        
        if self.pbar:
            pbar = tqdm(range(self.epoch))
        else:
            pbar = range(self.epoch)

        for epoch in pbar:
            for idx in range(self.n_weeks):
                start, end = 168 * idx, 168 * (idx + 1)
                X = self.train_X[start:end]
                Y = self.train_Y[start:end]
                # 前向传播
                output, _ = self.nn.forward(X)
                loss = loss_fn(output,Y)
                # 反向传播
                gd.zero_grad()
                loss.backward()
                gd.step()
                
            if self.pbar:
                pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch+1, loss))

    def _load(self, models:dict, file_path=None):
        for model in models.keys():
            try:
                print(f'load model in {file_path}')
                state = torch.load(file_path)
                models[model].load_state_dict(state[model])
            except FileNotFoundError:
                print(f'ERROR: model_saves file was not found in {file_path}, model not load.') 

    def _save(self, models:dict, file_path=None):
        if file_path is None:
            file_path = 'model_saves/' + self.name + '_models.pkl'
            print(f'use default value {file_path}.')

        state = {'epoch' : self.epoch}
        for model in models.keys():
            state[model] = models[model].state_dict()       
        
        try:
            torch.save(state, file_path)
        except FileNotFoundError:
            os.mkdir('model_saves/')
            torch.save(state, file_path)
        
        print(f'file saved in {file_path}.')

    def destroy(self):
        self.train_X = self.train_X.cpu().numpy()
        self.train_Y = self.train_Y.cpu().numpy()

        self.nn.cpu()
        torch.cuda.empty_cache()

class CloudServerDDNN:
    def __init__(self, name = "DDNN_global", batch_size=168, fs=[16,8,16], ks=3, n_weeks=3, n_his = 12):
        #local param
        stations = 5
        input_shape = [batch_size, 1, n_his, stations]
        #class param
        self.name = name
        self.n_weeks = n_weeks
        self.batch_size = batch_size

        #model nn
        self.nn = DCNN(input_shape, ks, fs, cls=10).cuda()

    def load_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._load({'nn':self.nn}, file_path)

    def save_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._save({'nn':self.nn}, file_path)

    def train(self, train_X:torch.Tensor=None, train_Y:torch.Tensor=None, epoch=500, pbar=True):
        self.nn.train()
        self.epoch = epoch
        self.pbar = pbar

        #training
        if train_X is not None:
            self.train_X = train_X
        if train_Y is not None:
            self.train_Y = train_Y

        self._training()
    
    def eval(self, test_X:torch.Tensor):
        self.nn.eval()
        test_X = test_X.float().cuda()
        
        return self.nn(test_X)

    def _training(self):
        #Optim & Lossfuc
        gd = torch.optim.Adam(self.nn.parameters())
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        
        if self.pbar:
            pbar = tqdm(range(self.epoch))
        else:
            pbar = range(self.epoch)

        for epoch in pbar:
            for idx in range(self.n_weeks):
                start, end = self.batch_size * idx, self.batch_size * (idx + 1)
                X = self.train_X[start:end]
                Y = self.train_Y[start:end]
                # 前向传播
                label = Y.contiguous().view(-1).long()
                output, _ = self.nn.forward(X)
                loss = loss_fn(output, label)
                # 反向传播
                gd.zero_grad()
                loss.backward()
                gd.step()
                
            if self.pbar:
                pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch+1, loss))

            with open('result.txt','a+') as f:
                f.write('loss:'+str(loss.data)+'\n')

    def _load(self, models:dict, file_path=None):
        for model in models.keys():
            try:
                print(f'load model in {file_path}')
                state = torch.load(file_path)
                models[model].load_state_dict(state[model])
            except FileNotFoundError:
                print(f'ERROR: model_saves file was not found in {file_path}, model not load.') 

    def _save(self, models:dict, file_path=None):
        if file_path is None:
            file_path = 'model_saves/' + self.name + '_models.pkl'
            print(f'use default value {file_path}.')

        state = {'epoch' : self.epoch}
        for model in models.keys():
            state[model] = models[model].state_dict()       
        
        try:
            torch.save(state, file_path)
        except FileNotFoundError:
            os.mkdir('model_saves/')
            torch.save(state, file_path)
        
        print(f'file saved in {file_path}.')

    def destroy(self):
        self.train_X = self.train_X.cpu().numpy()
        self.train_Y = self.train_Y.cpu().numpy()

        self.nn.cpu()
        torch.cuda.empty_cache()

def local_pred(Region = "D5"):
#Edge Server1
    routes_city = data_utils.station_gen_Aq(key=Region)
    test_X, test_Y = data_utils.data_gen(file_path = 'datasets/BeijingAq.csv', n_row=routes_city, n_days=7,
                                         n_his=n_his, offset=test_start,
                                         header=0, index_col='utc_time')
    test_X = torch.tensor(test_X)

    Edge_server = EdgeServerDDNN(name=Region, region=Region, n_weeks=train_weeks, n_his=n_his)
    if model_load:
        Edge_server.load_model()
    """ for i in tqdm(range(epoch)):
        Edge_server.train(epoch=1, pbar=False)

        _test_data = Edge_server.eval(test_X)[0]
        _test_data = _test_data.detach().cpu().numpy()
            
        static = math_utils.evaluation(test_Y,_test_data)
        with open('result.txt', 'a+') as f:
            for stat in static.keys():
                f.write(''+str(stat)+':'+str(static[stat])+'\n') """

    Edge_server.train(epoch=epoch)

    _test_data = Edge_server.eval(test_X)[0]
    _test_data = _test_data.detach().cpu().numpy()
            
    static = math_utils.evaluation(test_Y,_test_data)
    for stat in static.keys():
        print(str(stat)+':'+str(static[stat]))

    Edge_server.save_model()

#global
def global_pred():
    train_X, train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7 * train_weeks, n_his=n_his, offset=0,
                                             index_col='utc_time', header=0)
    test_X, test_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7, n_his=n_his, offset=test_start,
                                             index_col='utc_time', header=0)

    train_X = torch.tensor(train_X).float().cuda()
    train_Y = torch.tensor(train_Y).float().cuda()
    test_X = torch.tensor(test_X).float().cuda()
    test_Y = np.reshape(test_Y, (-1))

    #Cloud Server
    Bayarea = CloudServerDDNN(name="model_ddnn_bayarea" ,
                                batch_size = 168, fs=[16,8,16], ks=3,
                                n_weeks=train_weeks, n_his = n_his)
    if model_load:
        Bayarea.load_model()

    #Bayarea.train(train_X, train_Y, epoch=epoch, pbar=True)
    for i in range(epoch):
        Bayarea.train(train_X, train_Y, epoch=1, pbar=False)

        _test_data = np.argmax(Bayarea.eval(test_X)[0].detach().cpu().numpy(), 1)
        static = math_utils.evaluation(test_Y, _test_data, type='classification')

        with open('result.txt','a+') as f:
            for stat in static.keys():
                f.write(str(stat)+':'+str(static[stat])+'\n')



    _test_data = np.argmax(Bayarea.eval(test_X)[0].detach().cpu().numpy(), 1)

    static = math_utils.evaluation(test_Y, _test_data, type='classification')

    for stat in static.keys():
        print(str(stat)+':'+str(static[stat]))

    Bayarea.save_model()

if pred_type=='local':
    local_pred()

if pred_type=='global':
    global_pred()

