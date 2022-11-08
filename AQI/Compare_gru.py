import torch
from torch import nn
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

train_weeks = 30
test_start = 168 * train_weeks
n_his = 13

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

#local
if pred_type == 'local':

    routes = data_utils.station_gen_Aq(key=['D1','D2','D3','D4','D5'])
    n_node = len(routes)

    train_X, train_Y = data_utils.data_gen(file_path = 'datasets/BeijingAq.csv', n_row=routes, n_days=7 * train_weeks,
                                            n_his=n_his, offset=0,
                                            header=0, index_col='utc_time')
    test_X, test_Y = data_utils.data_gen(file_path = 'datasets/BeijingAq.csv', n_row=routes, n_days=7, 
                                            n_his=n_his, offset=test_start,
                                            index_col='utc_time', header=0)

    model = gru(input_size=n_his, hidden_size=16, output_size=1, num_layer=7, name='_gru_'+pred_type).cuda()
    criterion = nn.MSELoss()
#global
if  pred_type == 'global':
    train_X, train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7 * train_weeks, n_his=n_his, offset=0,
                                             index_col='utc_time', header=0)
    test_X, test_Y = data_utils.data_gen('datasets/BeijingMeo.csv', type='Weather', n_row=-5,
                                             n_days=7, n_his=n_his, offset=test_start,
                                             index_col='utc_time', header=0)

    model = gru(input_size=n_his, hidden_size=16, output_size=10, num_layer=7, name='_gru_'+pred_type).cuda()
    criterion = nn.CrossEntropyLoss()

train_X = np.transpose(np.squeeze(train_X), [0,2,1])
test_X = np.transpose(np.squeeze(test_X), [0,2,1])

train_X = torch.from_numpy(train_X).float().cuda()
train_Y = torch.from_numpy(train_Y).float().cuda()
test_X = torch.from_numpy(test_X).float().cuda()

if model_load:
    model.load()

optimizer = torch.optim.Adam(model.parameters())

# 开始训练
pbar = tqdm(range(epoch))
for e in pbar:
    for idx in range(train_weeks):
        start, end = 168 * idx, 168 * (idx+1)
        X = train_X[start: end]
        Y = train_Y[start: end]
        # 前向传播
        out = model(X).view(-1, 10)

        label = Y.contiguous().view(-1).long()
        loss = criterion(out, label)
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

#------------------------------------------------------------
pred_test = model(test_X).view(-1, 10)
pred_test = pred_test.cpu().detach().numpy()

if pred_type == 'local':
    static = math_utils.evaluation(test_Y, pred_test, type='regression')
if pred_type == 'global':
    pred_test = np.argmax(pred_test, 1)
    test_Y = np.reshape(test_Y, (-1))
    static = math_utils.evaluation(test_Y, pred_test, type='classification')
#------------------------------------------------------------

for stat in static.keys():
    print(str(stat)+':'+str(static[stat]))
    

