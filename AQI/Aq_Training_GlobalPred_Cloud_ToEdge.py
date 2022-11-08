import numpy as np
import torch
from tqdm import tqdm
from model_utils import global_params as gp
from model_utils import servers as sv
from model_utils import data_utils
from model_utils import math_utils

gp.__init()

#train_param
epoch = 500
n_weeks = 30
n_his = 12
fs = [16,8,16,8,16]

regions = ['D1','D2','D3','D4','D5']

train_X, train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather',
                                day_slot=24, n_his=n_his, n_days=7*n_weeks,
                                index_col='utc_time', header=0) 

test_X, test_Y = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather',
                                day_slot=24, n_his=n_his, n_days=7, offset=168*30,
                                index_col='utc_time', header=0)

train_X = torch.tensor(train_X)
train_Y = torch.tensor(train_Y)
test_X = torch.tensor(test_X)

#train Cloud_Server2
Beijing = sv.WqCloudServer(name = "Beijing_weather", Regions=regions,
                            batch_size=24*7, f_in=1, f_out=1, f_size=fs,
                            n_his=n_his, K_args=(3,1,1))
#Beijing.load_model()
#Beijing.train(train_X, train_Y, epochs=epoch)
for i in tqdm(range(epoch)):
    Beijing.train(train_X, train_Y, epochs=1)

    #Beijing.save_model()

    dtest_Y = np.argmax(Beijing.eval(test_X)[0].detach().cpu().numpy(), 2)

    #test_Y = np.transpose(test_Y)
    #dtest_Y = np.transpose(dtest_Y)

    test_Y = np.reshape(test_Y,(-1))
    dtest_Y = np.reshape(dtest_Y,(-1))

    #for i in range(len(dtest_Y)):
    static = math_utils.evaluation(dtest_Y,test_Y,'classification')
        
    with open('result.txt','a+') as f:
        for stat in static.keys():
            f.write(str(stat)+':'+str(static[stat])+'\n')

from model_utils import plot
#plot.plot_weather_acc(np.clip(abs(dtest_Y-test_Y),0,1))