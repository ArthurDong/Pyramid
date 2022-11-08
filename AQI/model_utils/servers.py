import os
import torch
import numpy as np
from tqdm import tqdm
from model_utils import global_params as gp
from model_utils import layers
from model_utils import data_utils

class Server:
    def __init__(self, name:str, region:str or list, epochs:int, c_in:int, feature_size_list:list, ksargs:tuple, day_slot:int, n_days:int, n_weeks:int, n_his = 21):
        #local param
        station = data_utils.station_gen_Aq(key=region)
        Tensor_shape = [day_slot * n_days, c_in, n_his, len(station)]
        #class param
        self.name = name
        self.day_slot = day_slot
        self.epochs = epochs
        self.n_days = n_days
        self.n_weeks = n_weeks
        #kernel update
        train_weight = data_utils.weight_gen(gp.AdjacentGraph, station, ksargs[1])
        gp.update(train_weight, None)
        #model
        self.Pyramid = layers.Pyramid(Tensor_shape, [0,0,0,station.size], feature_size_list, ksargs[0], ksargs[1], ksargs[2], mode='local').cuda()
        #train, test data
        self.train_X, self.train_Y = data_utils.data_gen('datasets/BeijingAq.csv', 
                                                            n_row = station, loop = True,
                                                            day_slot = day_slot, n_his = n_his,
                                                            n_days = n_weeks*n_days, 
                                                            index_col = 'utc_time', header=0)

    def load_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._load({'hiernn':self.Pyramid}, file_path)

    def save_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._save({'hiernn':self.Pyramid}, file_path)

    def train(self, train_data=None, test_data=None, epochs=500):
        #set epoch
        self.epochs = epochs
        #training
        if train_data is None and test_data is None:
            self._training(self.train_X, self.train_Y)
        elif train_data is None:
            self._training(self.train_X, test_data)
        elif test_data is None:
            self._training(train_data, self.train_Y)
        else:
            self._training(train_data, test_data)

    def eval(self, test_data):
        test_data = torch.tensor(test_data).float().cuda()
        output, aux_output = self.Pyramid(test_data)
        test_data.cpu()
        output = output.detach().cpu().numpy()
        aux_output = aux_output.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return output, aux_output

    def _training(self, train_data, train_label):
        #Optim & Lossfuc
        gd = torch.optim.Adam(self.Pyramid.parameters())
        loss_fn = torch.nn.MSELoss()

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            for idx in range(self.n_weeks):
                start, end = self.day_slot * self.n_days * idx, self.day_slot * self.n_days * (idx + 1)
                X = torch.tensor(train_data[start:end]).float().cuda()
                Y = torch.tensor(train_label[start:end]).float().cuda()

                gd.zero_grad()
                output, _ = self.Pyramid.forward(X)
                loss = loss_fn(output,Y)
                loss.backward()
                gd.step()

                pbar.set_description("train loss: %.6f"%loss)
        
                #X.cpu()
                #Y.cpu()
        #torch.cuda.empty_cache()
    
    def _load(self, models:dict, file_path:str):
        for model in models.keys():
            try:
                print(f'load model in {file_path}')
                state = torch.load(file_path)
                models[model].load_state_dict(state[model])
                gp.set_max_value(state['max_value'])
            except FileNotFoundError:
                print(f'ERROR: model_saves file was not found in {file_path}, model not load.') 

    def _save(self, models:dict, file_path:str):
        state = {'epoch' : self.epochs}
        for model in models.keys():
            state[model] = models[model].state_dict()  
        state['max_value'] = gp.max_value    
        
        try:
            torch.save(state, file_path)
        except FileNotFoundError:
            os.mkdir('model_saves/')
            torch.save(state, file_path)
        
        print(f'model saved in {file_path}.')

""" class AqEdgeServer(Server):
    def __init__(self, name = "D1_", Region="D1",
                epochs=500, c_in=1, c_out=[16,8,16], args=(3,1,0), day_slot=24, n_days=7, n_weeks=3, n_his = 21):
        super().__init__(name=name, region=Region, epochs=epochs, c_in=c_in, feature_size_list=c_out, ksargs=args, day_slot=day_slot, n_days=n_days, n_weeks=n_weeks, n_his = n_his)
 """
class AqCloudServer(Server):
    def __init__(self, name = "Beijing_weather", Regions=["D1","D2","D3","D4","D5"],
                epochs=500, c_in =4, c_out=[16,8,16], args=(3,1,1), day_slot=24, n_days=7, n_weeks=3, n_his = 21):
        super().__init__(name=name, region=Regions, epochs=epochs, c_in=c_in, feature_size_list=c_out, ksargs=args, day_slot=day_slot, n_days=n_days, n_weeks=n_weeks, n_his = n_his)
        n_node = data_utils.station_gen_Aq(key=Regions).size
        n_node_out = len(Regions)
        Tensor_shape = [day_slot * n_days, c_in, n_his, n_node]
        #kernel update
        trans_matrix= data_utils.transformMetric_gen_Aq() 
        train_weight = data_utils.weight_gen_cloud_Aq()
        gp.update(train_weight, trans_matrix)

        self.HierNN = layers.AqGLN(Tensor_shape, args[0], args[1], args[2], c_out, n_node_out).cuda()

    def _training(self, train_data, train_label):
            #Optim & Lossfuc
            gd = torch.optim.Adam(self.HierNN.parameters(),lr=10e-2)
            loss_fn = torch.nn.CrossEntropyLoss().cuda()

            pbar = tqdm(range(self.epochs))
            for epoch in pbar:
                for idx in range(self.n_weeks):
                    start, end = self.day_slot * self.n_days * idx, self.day_slot * self.n_days * (idx + 1)
                    X = torch.tensor(train_data[start:end]).float().cuda()
                    Y = torch.tensor(train_label[start:end]).int().cuda()

                    gd.zero_grad()
                    output, _ = self.HierNN(X)

                    label = Y.view(-1).long()
                    output = output.reshape(-1, output.shape[2])

                    loss = loss_fn(output, label).cpu()
                    loss.backward()
                    gd.step()
                    
                    pbar.set_description("train loss: %.5f"%loss)

                    X.cpu()
                    Y.cpu()

class WqCloudServer:
    def __init__(self, name = "Beijing_weather", Regions=["D1","D2","D3","D4","D5"],
                    batch_size = 24*7, f_in=1, f_out=1, f_size=[16,8,16], n_his=23, K_args=(3,1,1)):

        self.name = name
        self.batch_size = batch_size

        Tensor_in = [batch_size, f_in, n_his, len(Regions)]
        Tensor_out = [batch_size, f_out, -1, len(Regions)]
        #kernel update
        train_weight = data_utils.weight_gen_cloud_Aq()
        gp.update(train_weight, None)

        self.PyramidAQ = layers.Pyramid(shape_in=Tensor_in, shape_out=Tensor_out, feature_size_list=f_size, Kt=K_args[0], Ks=K_args[1], Kr=K_args[2], mode='global').cuda()

    def train(self, train_X:torch.Tensor, train_Y:torch.Tensor, epochs=500):
        #set epoch
        self.epochs = epochs
        #training
        gd = torch.optim.Adam(self.PyramidAQ.parameters())
        loss_fn = torch.nn.CrossEntropyLoss().cuda()

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for idx in range(train_X.shape[0]//self.batch_size):
                start, end = self.batch_size * idx, self.batch_size * (idx + 1)

                X = train_X[start:end].float().cuda()
                Y = train_Y[start:end].int().cuda()

                gd.zero_grad()
                output = self.PyramidAQ.forward(X)[0]
                output = output.reshape(-1, output.shape[2])
                label = Y.contiguous().view(-1).long()

                loss = loss_fn(output, label)
                loss.backward()
                gd.step()
                
            pbar.set_description("epoch: %d, train loss: %.5f"%(epoch, loss))

            with open('result.txt','a+') as f:
                f.write('loss:'+str(loss.data)+'\n')

    def eval(self, test_X:torch.Tensor):
        test_X = test_X.float().cuda()
        return self.PyramidAQ.forward(test_X)

class AqEdgeServer:
    def __init__(self, name = "D1_air_quality", Region='D1',
                    batch_size = 24*7, f_in=1, f_out=1, f_size=[32,16,8,8,4], n_his=23, K_args=(3,1,1)):

        self.name = name
        self.batch_size = batch_size

        Tensor_in = [batch_size, f_in, n_his, len(data_utils.station_gen_Aq(key=Region))+1]
        Tensor_out = [batch_size, f_out, -1, len(data_utils.station_gen_Aq(key=Region))]
        #kernel update
        train_weight = data_utils.weight_gen(gp.AdjacentGraph, data_utils.station_gen_Aq(key=Region), K_args[1])
        trans_metric = np.ones([len(data_utils.station_gen_Aq(key=Region))+1, len(data_utils.station_gen_Aq(key=Region))])
        gp.update(train_weight, trans_metric)

        self.PyramidAQ = layers.Pyramid(shape_in=Tensor_in, shape_out=Tensor_out, feature_size_list=f_size, Kt=K_args[0], Ks=K_args[1], Kr=K_args[2], mode='local').cuda()

    def train(self, train_X:torch.Tensor, train_Y:torch.Tensor, epochs=500):
        #set epoch
        self.epochs = epochs
        #training
        gd = torch.optim.Adam(self.PyramidAQ.parameters())
        loss_fn = torch.nn.MSELoss().cuda()

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            for idx in range(train_X.shape[0]//self.batch_size):
                start, end = self.batch_size * idx, self.batch_size * (idx + 1)

                X = train_X[start:end].float().cuda()
                Y = train_Y[start:end].float().cuda()

                gd.zero_grad()
                output = self.PyramidAQ(X)[0]

                loss = loss_fn(output, Y)
                loss.backward()
                gd.step()
                
                pbar.set_description("epoch: %d, train loss: %.5f"%(epoch, loss))

    def eval(self, test_X:torch.Tensor):
        test_X = test_X.float().cuda()
        return self.PyramidAQ.forward(test_X)
