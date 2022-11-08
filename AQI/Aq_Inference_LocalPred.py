import numpy as np
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.math_utils as math_utils
import model_utils.global_params as gp
from model_utils import plot
import random

#eval_param
n_days = 7
Region = 'D3'
n_his = 12

gp.__init()
train_data, test_data = data_utils.data_gen('datasets/BeijingAq.csv', loop=True, n_his=n_his,
                                            day_slot=24, n_days=n_days,
                                            index_col='utc_time', header=0)

station_Beijing = data_utils.station_gen_Aq(key=Region)

Beijing = sv.AqEdgeServer(name="AQI_"+Region ,Region=Region, n_weeks=1, c_out=[32,16,64,8,16], n_his = n_his)
Beijing.load_model()

#Cloud pred
evalBeijing, _ = Beijing.eval(train_data[:,:,:,station_Beijing])
evalBeijing *= gp.max_value
#Orignal data
_evalBeijing = test_data[:,station_Beijing]*gp.max_value

#evaluate routes
print("----Edge Server of Bayarea----")
static = math_utils.evaluation(evalBeijing,_evalBeijing)
for stat in static.keys():
    res = ''+str(stat)+':'+str(static[stat])
    print(res)

seed = random.randint(0,evalBeijing.shape[1]-1)

plot.plot_AQI((_evalBeijing[:,seed],evalBeijing[:,seed]),"fig/"+'station')