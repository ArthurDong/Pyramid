import numpy as np
import model_utils.global_params as gp
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import time

gp.__init()

#train_param
epoch = 500
n_weeks = 30
n_his = 12
c_out = [32,16,64,8,16]

day_slot = 24
regions = ['D1','D2','D3','D4','D5']

#Edge Server1
Distric1 = sv.AqEdgeServer(name="AQI_D1", Region="D1", n_weeks=n_weeks, c_out=c_out, n_his=n_his)
Distric1.load_model()

#Edge Server2
Distric2 = sv.AqEdgeServer(name="AQI_D2", Region="D2", n_weeks=n_weeks, c_out=c_out, n_his=n_his)
Distric2.load_model()

#Edge Server3
Distric3 = sv.AqEdgeServer(name="AQI_D3", Region="D3", n_weeks=n_weeks, c_out=c_out, n_his=n_his)
Distric3.load_model()

#Edge Server4
Distric4 = sv.AqEdgeServer(name="AQI_D4", Region="D4", n_weeks=n_weeks, c_out=c_out, n_his=n_his)
Distric4.load_model()

#Edge Server5
Distric5 = sv.AqEdgeServer(name="AQI_D5", Region="D5", n_weeks=n_weeks, c_out=c_out, n_his=n_his)
Distric5.load_model()

#prepare Cloud Servers' dataset
train_data, _ = data_utils.data_gen('datasets/BeijingAq.csv',
                                    day_slot=24, n_his=n_his, n_days=7*n_weeks,
                                    index_col='utc_time', header=0)

indexD1 = data_utils.station_gen_Aq(key="D1")
indexD2 = data_utils.station_gen_Aq(key="D2")
indexD3 = data_utils.station_gen_Aq(key="D3")
indexD4 = data_utils.station_gen_Aq(key="D4")
indexD5 = data_utils.station_gen_Aq(key="D5")

#concatenate Edge Servers' dataset
earlyExits = []
for idx in range(n_weeks):
    start, end = day_slot * 7 * idx, day_slot * 7 * (idx + 1)
    F_out = []
    F_out.append(Distric1.eval(train_data[start:end,:,:,indexD1])[1])
    F_out.append(Distric2.eval(train_data[start:end,:,:,indexD2])[1])
    F_out.append(Distric3.eval(train_data[start:end,:,:,indexD3])[1])
    F_out.append(Distric4.eval(train_data[start:end,:,:,indexD4])[1])
    F_out.append(Distric5.eval(train_data[start:end,:,:,indexD5])[1])
    earlyExits.append(np.concatenate(F_out, axis=3))

train_X = np.concatenate(earlyExits) 

train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather',
                                day_slot=24, n_his=1, n_days=7*n_weeks,
                                index_col='utc_time', header=0)[1] 

train_X, _ = data_utils.data_gen('datasets/BeijingAq.csv', type='Aqi',
                                day_slot=24, n_his=5, n_days=7*n_weeks, n_row=data_utils.station_gen_Aq(key=regions),
                                index_col='utc_time', header=0)

_, train_Y = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather',
                                day_slot=24, n_his=1, n_days=7*n_weeks,
                                index_col='utc_time', header=0)


#train Cloud_Server2
Beijing = sv.AqCloudServer(name="Weather_Beijing", Regions=regions, c_in=1, c_out=[], n_weeks=n_weeks, n_days=7, n_his=5)
#Beijing.load_model()
Beijing.train(train_X, train_Y, epochs=epoch)
Beijing.save_model()

from model_utils import math_utils
test_X = train_X[:168]
dtest_Y = np.argmax(Beijing.eval(test_X)[0], 2)
test_Y = train_Y[:168]

test_Y = np.transpose(test_Y)
dtest_Y = np.transpose(dtest_Y)

for i in range(len(dtest_Y)):
    static = math_utils.evaluation(dtest_Y[i],test_Y[i],'classification')

    for stat in static.keys():
        res = ''+str(stat)+':'+str(static[stat])
        print(res)

print(dtest_Y[4,60:120],test_Y[4,60:120])


from model_utils import plot
plot.plot_weather_acc(np.clip(abs(dtest_Y-test_Y),0,1))