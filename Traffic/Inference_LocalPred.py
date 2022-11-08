
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.math_utils as math_utils
import model_utils.global_params as gp

import torch
import numpy as np

#eval_param
n_days = 7
n_weeks = 1
n_his = 13
CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]

gp.__init()

#import test instances
train_data, test_data = data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=n_days*n_weeks, n_his=n_his, offset=3024)
routes_BayArea = data_utils.station_gen(key=CityList)

#Edge Server1
Fremont = sv.Server(city="City of Fremont", n_weeks=n_weeks, n_his=n_his)
Fremont.load_model()

#Edge Server2
Oakland = sv.Server(city="City of Oakland", n_weeks=n_weeks, n_his=n_his)
Oakland.load_model()

#Edge Server3
Richmond = sv.Server(city="City of Richmond", n_weeks=n_weeks, n_his=n_his)
Richmond.load_model()

#Edge Server4
SanFrancisco = sv.Server(city="City of San Francisco", n_weeks=n_weeks, n_his=n_his)
SanFrancisco.load_model()

#Edge Server5
SanJose = sv.Server(city="City of San Jose", n_weeks=n_weeks, n_his=n_his)
SanJose.load_model()

#prepare Cloud Servers' dataset
train_data = torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=n_days*n_weeks, n_his=n_his, offset=3024)[0])
routes_Fremont = data_utils.station_gen(key="City of Fremont")
routes_Oakland = data_utils.station_gen(key="City of Oakland")
routes_Richmond = data_utils.station_gen(key="City of Richmond")
routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
routes_SanJose = data_utils.station_gen(key="City of San Jose")

#Edge pred
earlyExitList, outputList = [],[]
F_out = Fremont.eval(train_data[:,:,:,routes_Fremont])
earlyExitList.append(F_out[0])
outputList.append(F_out[1])
F_out = Oakland.eval(train_data[:,:,:,routes_Oakland])
earlyExitList.append(F_out[0])
outputList.append(F_out[1])
F_out = Richmond.eval(train_data[:,:,:,routes_Richmond])
earlyExitList.append(F_out[0])
outputList.append(F_out[1])
F_out = SanFrancisco.eval(train_data[:,:,:,routes_SanFrancisco])
earlyExitList.append(F_out[0])
outputList.append(F_out[1])
F_out = SanJose.eval(train_data[:,:,:,routes_SanJose])
earlyExitList.append(F_out[0])
outputList.append(F_out[1])

del F_out

outputEdge = torch.cat(outputList, axis=3).detach().cpu().numpy()

#Edge HierML
earlyExitEdge = torch.cat(earlyExitList, axis=1).detach().cpu().numpy()

#Orignal Data
_outputBayArea = test_data[:,routes_BayArea]

#draw figure
seed = 110
original = _outputBayArea[:,seed]
Hierml = earlyExitEdge[:,seed]

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#plt.style.use('ggplot')
plt.figure(figsize=(7.9,6.5))

plt.xticks(range(0,1009,144),range(0,8,1), fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel('Day', fontsize=25)
plt.ylabel('Traffic Flow', fontsize=25)
plt.grid(linestyle='-.', linewidth=1.5)

plt.plot(original*1000, color='dodgerblue', linestyle='-', label = 'Ground Truth')
plt.plot(Hierml*1000, color='red', linestyle='-', label = 'Pyramid-TP')
plt.legend(ncol=2, fontsize=25, bbox_to_anchor=(0.5, 1), loc=3, borderaxespad=0.35)

plt.savefig('./fig/LocalPred.png')
plt.close()