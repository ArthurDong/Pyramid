import numpy as np
import model_utils.global_params as gp
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.math_utils as math_utils
import random
import torch

gp.__init()

#eval_param
n_days = 7
n_weeks = 1
n_his = 13
CityList = ["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"]

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

#Cloud_Server_GlobalPred
roadDict, _outputBayArea = data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_days=n_weeks*n_days, n_his=n_his)
BayArea = sv.CloudServer(city=CityList, dataDict = roadDict, n_weeks=n_weeks, n_days=n_days, n_his=n_his)
BayArea.load_model()

#prepare Cloud Servers' dataset
train_data = torch.tensor(data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=n_days*n_weeks, n_his=n_his)[0])
routes_Fremont = data_utils.station_gen(key="City of Fremont")
routes_Oakland = data_utils.station_gen(key="City of Oakland")
routes_Richmond = data_utils.station_gen(key="City of Richmond")
routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
routes_SanJose = data_utils.station_gen(key="City of San Jose")

#Edge pred
outputList = []
outputList.append(Fremont.eval(train_data[:,:,:,routes_Fremont])[1])
outputList.append(Oakland.eval(train_data[:,:,:,routes_Oakland])[1])
outputList.append(Richmond.eval(train_data[:,:,:,routes_Richmond])[1])
outputList.append(SanFrancisco.eval(train_data[:,:,:,routes_SanFrancisco])[1])
outputList.append(SanJose.eval(train_data[:,:,:,routes_SanJose])[1])

outputEdge = torch.cat(outputList, axis=3)

#Cloud pred
outputBayArea = BayArea.eval(outputEdge)[0].detach().cpu().numpy()

#Orignal data
roadDict, _outputBayArea = data_utils.highway_data_generator(gp.AdjacentGraph, cityList=CityList, n_weeks=n_weeks*n_days)

#evaluate routes
print("----Cloud Server of Bay Area----")
static = math_utils.evaluation(_outputBayArea, outputBayArea)
for stat in static.keys():
    print(str(stat)+':'+str(static[stat]))


def plot_name(roadDict, id=0):
    cityId = None
    for City in iter(list(roadDict.keys())):
        if id >len(roadDict[City]):
            cityId = City
            id -= len(roadDict[City]) 
        else:
            break

    cities = str(cityId).split(" / ")
    city1, city2  = cities[0], cities[1]
    return city1 + "_" + city2 + "_" + roadDict[cityId][id]

#draw figure
seed = random.randint(0,outputBayArea.shape[1]-1)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#plt.style.use('ggplot')
plt.figure(figsize=(7.9,6.5))

plt.xticks(range(0,1009,144),range(0,8,1), fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel('Day', fontsize=25)
plt.ylabel('Traffic Flow', fontsize=25)
plt.grid(linestyle='-.', linewidth=1.5)

plt.plot(_outputBayArea[:,seed]*1000, color='dodgerblue', linestyle='-', label = 'Ground Truth')
plt.plot(outputBayArea[:,seed]*1000, color='red', linestyle='-', label = 'Pyramid-TP')
plt.legend(ncol=2, fontsize=25, bbox_to_anchor=(-0.8, 1), loc=3, borderaxespad=0.35)

plt.savefig("./fig/GlobalPred.png")
plt.close()