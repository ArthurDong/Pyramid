import numpy as np
import model_utils.global_params as gp
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.math_utils as math_utils
import random
import time

gp.__init()

#train_param
n_weeks = 1
n_days = 7
day_slot = 24
regions = ['D1','D2','D3','D4','D5']
c_out = [32,16,64,8,16]
n_his = 12

#Edge Server1
Distric1 = sv.AqEdgeServer(name="AQI_D1", Region="D1", n_weeks=n_weeks, n_his=n_his, c_out=c_out)
Distric1.load_model()

#Edge Server2
Distric2 = sv.AqEdgeServer(name="AQI_D2", Region="D2", n_weeks=n_weeks, n_his=n_his, c_out=c_out)
Distric2.load_model()

#Edge Server3
Distric3 = sv.AqEdgeServer(name="AQI_D3", Region="D3", n_weeks=n_weeks, n_his=n_his, c_out=c_out)
Distric3.load_model()

#Edge Server4
Distric4 = sv.AqEdgeServer(name="AQI_D4", Region="D4", n_weeks=n_weeks, n_his=n_his, c_out=c_out)
Distric4.load_model()

#Edge Server5
Distric5 = sv.AqEdgeServer(name="AQI_D5", Region="D5", n_weeks=n_weeks, n_his=n_his, c_out=c_out)
Distric5.load_model()

#Cloud_Server_Task2
Beijing = sv.AqCloudServer(name="Weather_Beijing", Regions=regions, n_weeks=n_weeks, n_days=n_days)
Beijing.load_model()

#prepare Cloud Servers' dataset
train_data, _ = data_utils.data_gen('datasets/BeijingAq.csv', loop=True, 
                                    day_slot=24, n_his=n_his, n_days=n_days*n_weeks,
                                    index_col='utc_time', header=0)

indexD1 = data_utils.station_gen_Aq(key="D1")
indexD2 = data_utils.station_gen_Aq(key="D2")
indexD3 = data_utils.station_gen_Aq(key="D3")
indexD4 = data_utils.station_gen_Aq(key="D4")
indexD5 = data_utils.station_gen_Aq(key="D5")

#concatenate Edge Servers' dataset
earlyExits = []
for idx in range(n_weeks):
    start, end = day_slot * n_days * idx, day_slot * n_days * (idx + 1)
    F_out = []
    F_out.append(Distric1.eval(train_data[start:end,:,:,indexD1])[1])
    F_out.append(Distric2.eval(train_data[start:end,:,:,indexD2])[1])
    F_out.append(Distric3.eval(train_data[start:end,:,:,indexD3])[1])
    F_out.append(Distric4.eval(train_data[start:end,:,:,indexD4])[1])
    F_out.append(Distric5.eval(train_data[start:end,:,:,indexD5])[1])
    earlyExits.append(np.concatenate(F_out, axis=3))

train_data_cloud = np.concatenate(earlyExits)
_, test_data_cloud = data_utils.data_gen('datasets/BeijingMeo.csv', n_row=-5, type='Weather', loop=True, 
                                    day_slot=24, n_his=1, n_days=n_days*n_weeks,
                                    index_col='utc_time', header=0)

#CloudServer Evaluation

eval_Beijing, _ = Beijing.eval(train_data_cloud)
eval_Beijing = np.argmax(eval_Beijing, 2)

#evaluate routes
print("----Cloud Server of Bayarea----")
seed = random.randint(0,eval_Beijing.shape[1]-1)
math_utils.evaluation(eval_Beijing[:,seed],test_data_cloud[:,seed],'classification')

#draw figure
import matplotlib.pyplot as plt
import os
if not os.path.exists('fig'):
    os.mkdir('fig')

plt.switch_backend('agg')
#plt.title('region_'+str(seed), bbox=dict(facecolor='gray', edgecolor='black', alpha=0.65))
plt.xlabel('Time slots', fontsize=15)
plt.ylabel('Weather_class', fontsize=15)
plt.xticks([24,48,72,96,120,144,168],[1,2,3,4,5,6,7])

plt.plot(test_data_cloud[:,seed],label = 'Original Data')
plt.plot(eval_Beijing[:,seed], label = 'Prediction Edge')

plt.legend()
plt.savefig("fig/"+'region_'+str(seed))
plt.close()