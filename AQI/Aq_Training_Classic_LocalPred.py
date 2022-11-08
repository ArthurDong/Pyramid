import numpy as np
import model_utils.servers as sv
import model_utils.global_params as gp
import model_utils.data_utils as data_utils
import time

#train
n_weeks = 3
n_days = 7
epoch = 300
city = ['D1','D2','D3','D4','D5']

gp.__init(1)

#Cloud Server
st = time.time()
Beijing = sv.Server(name="BeijingAq" ,region=city, epochs=epoch, n_weeks=n_weeks, feature_size_list=[128,16,32,4,4])
#Beijing.load_model()
Beijing.train()
st2 = time.time()
print("training time BeijingAQI %.2fs"%(st2 - st)) 

#eval
import random
import model_utils.math_utils as math_utils
n_days = 7
n_weeks = 1

#import test instances
train_data, test_data = data_utils.data_gen('datasets/Beijing_dataset/BeijingAq.csv', loop=True, day_slot=24, n_days=n_days*n_weeks, index_col='utc_time', header=0)
station_Beijing = data_utils.station_gen_Aq(key=city)

size = train_data.nbytes/1024/1024
print('Matrix size:',size,'MB') 

#Cloud pred
evalBeijing, _ = Beijing.eval(train_data[:,:,:,station_Beijing])

#Orignal data
_evalBeijing = test_data[:,station_Beijing]

#evaluate routes
print("----Edge Server of Bayarea----")
seed = random.randint(0,evalBeijing.shape[1]-1)
math_utils.evaluation(evalBeijing[:,seed],_evalBeijing[:,seed])
