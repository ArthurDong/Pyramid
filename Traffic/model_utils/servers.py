from model_utils import global_params as gp
from model_utils import layers
from model_utils import data_utils
from tqdm import tqdm

import torch
import os

class Server:
    def __init__(self, name=None, city="City of Fremont", epoch=500, c_out=[16,8,16,8,16], args=(3,1), n_days=7, n_weeks=3, n_his=21):
        #local param
        routes = data_utils.station_gen(key=city)
        Tensor_shape = [144 * n_days, 1, n_his, routes.size]
        #class param
        if name == None:
            self.name = city
        else:
            self.name = name
        self.epoch = epoch
        self.n_days = n_days
        self.n_weeks = n_weeks
        self.n_route = routes
        #kernel update
        train_weight = data_utils.weight_gen(gp.AdjacentGraph, routes, args[1])
        gp.update(train_weight, None)
        #model nn
        self.nn = layers.STGCN(Tensor_shape, args[0], args[1], c_out).cuda()
        #train, test data
        self.train_X, self.train_Y = data_utils.data_gen('datasets/PemsD4.csv', n_route=self.n_route, loop=True, n_days=self.n_weeks*self.n_days, n_his=n_his)

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
                start, end = 1008 * idx, 1008 * (idx + 1)
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


class EdgeServer(Server):
    def __init__(self, name = None, city="City of Fremont", epoch=500, c_out=[16,8,16,8,16], args=(3,1), n_days=7, n_weeks=3, n_his = 21):
        Server.__init__(self, name=name, city=city, epoch=epoch, c_out=c_out, args=args, n_days=n_days, n_weeks=n_weeks, n_his=n_his)

class EdgeServerDDNN(Server):
    def __init__(self, name = "ddnn_local", city="City of Fremont", epoch=500, c_out=[16,8,16,8,16], Kernel_size=3, n_days=7, n_weeks=3, n_his = 21):
        name = 'model_ddnn_local_'+name
        Server.__init__(self, name=name, city=city, epoch=epoch, c_out=c_out, n_days=n_days, n_weeks=n_weeks, n_his=n_his)
        routes = data_utils.station_gen(key=city)
        Tensor_shape = [144 * n_days, 1, n_his, routes.size]
        self.nn = layers.DCNN(Tensor_shape, Kernel_size, c_out).cuda()

class CloudServerDDNN(Server):
    def __init__(self, name = 'ddnn_global', city=["City of Fremont","City of San Jose"], epoch=500, c_out=[16,8,16], Kernel_size=3, n_days=7, n_weeks=3, n_his = 15):
        Server.__init__(self, name=name, city=city, epoch=epoch, c_out=c_out, n_days=n_days, n_weeks=n_weeks, n_his = n_his)
        #Tensor_shape = list(X.shape) #[144 * n_days * n_weeks, 1, 21, routes.size] -> [144 * n_days, 1, 21, routes.size]
        routes = data_utils.station_gen(key=city)
        Tensor_shape = [144 * n_days, 16,  n_his, routes.size]
        self.nn = layers.DCNN(Tensor_shape, Kernel_size, c_out).cuda()

class CloudServer(Server):
    def __init__(self,name = 'BayArea_global', city=["City of Fremont","City of San Jose"], dataDict = {"City of Fremont / City of San Jose":['I880-N', 'I880-S', 'I680-S']}, epoch=500, c_out=[8,4,8], args=(3,1), n_days=7,  n_his=15, n_weeks=3):
        Server.__init__(self, name=name, city=city, epoch=epoch, c_out=c_out, args=args, n_days=n_days, n_weeks=n_weeks,  n_his=n_his)
        trans_matrix= data_utils.transformMetric_gen(dataDict, city)
        train_weight = data_utils.weight_gen_cloud(dataDict)
        #kernel update
        gp.update(train_weight, trans_matrix)

        #TensorShape = [144 * n_days * n_weeks, 1, 21, routes.size] -> [144 * n_days, 1, 21, routes.size]
        routes = data_utils.station_gen(key=city)
        t_out = train_weight.shape[0]

        Tensor_shape = [144 * n_days, 16, n_his-6, routes.size]
        self.extract = layers.TransformCov(Tensor_shape, K_r=3, c_out=t_out).cuda()
        
        #TensorShape = [144 * n_days, 1, 21, routes.size] -> [144 * n_days, 1, 21, t_out]
        Tensor_shape = [144 * n_days, 16, n_his-6, t_out]
        self.nn = layers.STGCN(Tensor_shape, args[0], args[1], c_out).cuda()

    def load_model(self):
        #load model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        self._load({'nn':self.nn,'extract':self.extract}, file_path)

    def save_model(self):
        #save model
        file_path = 'model_saves/' + self.name + '_models.pkl'
        #save model
        self._save({'nn':self.nn,'extract':self.extract}, file_path=file_path)
        
    def _training(self):
            #Optim & Lossfuc
            gd = torch.optim.Adam([{'params':self.nn.parameters()}, {'params':self.extract.parameters(),'lr':1e-3}], lr=1e-3)
            loss_fn = torch.nn.MSELoss().cuda()

            if self.pbar:
                pbar = tqdm(range(self.epoch))
            else:
                pbar = range(self.epoch)

            for epoch in pbar:
                for idx in range(self.n_weeks):
                    start, end = 144 * self.n_days * idx, 144 * self.n_days * (idx + 1)
                    X = self.train_X[start:end]
                    Y = self.train_Y[start:end]

                    gd.zero_grad()
                    M = self.extract.forward(X)
                    output, _ = self.nn.forward(M)
                    loss = loss_fn(output,Y)
                    loss.backward()
                    gd.step()

                    if self.pbar:
                        pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch+1, loss))

            X.cpu()
            Y.cpu()

    def eval(self, test_X:torch.Tensor):
        self.extract.eval()
        self.nn.eval()

        test_X = test_X.float().cuda()

        return self.nn(self.extract(test_X))