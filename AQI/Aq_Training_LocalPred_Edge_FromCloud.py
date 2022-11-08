import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.global_params as gp
import model_utils.math_utils as math_utils
import torch

#train
n_weeks = 13
test_start = 168 * (n_weeks)
epoch = 500
n_his = 16

gp.__init()

#Get train_X
data_X, _ = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather',
                                day_slot=24, n_his=n_his, n_days=7*(n_weeks+1),
                                index_col='utc_time', header=0) 

data_X = torch.tensor(data_X[:168*(n_weeks+1)])

Beijing = sv.WqCloudServer(name = "Beijing_weather", Regions=['D1','D2','D3','D4','D5'],
                            batch_size=24*7, f_in=1, f_out=1, f_size=[16,8,4],
                            n_his=n_his, K_args=(3,1,1))
tmp = []
for idx in range(n_weeks+1):
    start, end = 168 * idx, 168 * (idx + 1)
    tmp.append(Beijing.eval(data_X[start:end])[1])
data_X = torch.cat(tmp).detach()[:,:,:,0].unsqueeze(3).cpu()

data_X_2, data_Y = data_utils.data_gen('datasets/BeijingAq.csv', n_row=data_utils.station_gen_Aq(key='D1'),
                                day_slot=24, n_his=n_his-4, n_days=7*(n_weeks+1), type='Aqi',
                                index_col='utc_time', header=0) 

data_X_2 = torch.tensor(data_X_2).repeat((1,4,1,1))
data_X_A = torch.cat((data_X, data_X_2), dim=3)
#data_X_A = data_X_2

train_X = torch.tensor(data_X_A[:168*n_weeks])
test_X = torch.tensor(data_X_A[:168])

train_Y = torch.tensor(data_Y[:168*n_weeks])
test_Y = data_Y[:168]

AQ_Edge = sv.AqEdgeServer(name='D1_air_quality' , Region='D1',
                            batch_size=24*7, f_in=4, f_out=1, f_size=[32,16,64,8,16],
                            n_his=n_his-4, K_args=(3,1,1))
#AQ_Edge.load_model()
AQ_Edge.train(train_X, train_Y, epochs=epoch)
#AQ_Edge.save_model()

d_test_Y = AQ_Edge.eval(test_X)[0].detach().cpu().numpy()
dtest_Y = test_Y

static = math_utils.evaluation(dtest_Y,d_test_Y)

for stat in static.keys():
    res = ''+str(stat)+':'+str(static[stat])
    print(res)

import random
from model_utils import plot
seed = random.randint(0, len(data_utils.station_gen_Aq(key='D1')))
#plot.plot_AQI((dtest_Y[:,seed],d_test_Y[:,seed]),"fig/"+'station')