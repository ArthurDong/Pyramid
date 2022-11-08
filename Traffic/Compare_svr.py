import numpy as np
from sklearn import svm
from model_utils import math_utils
from model_utils import data_utils
from tqdm import tqdm

Prediction  =  'local'

if Prediction == 'global':
    data_x, data_y = data_utils.data_gen(file_path='datasets/PemsD4_cloud.csv', n_days=7, n_his=1, loop=True, offset=302)

if Prediction  ==  'local':
    routes = data_utils.station_gen(key=["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"])
    data_x, data_y = data_utils.data_gen(file_path = 'datasets/PemsD4.csv', n_route=routes, n_days=7, n_his=1, loop=True, offset=3024)

data_x = np.squeeze(data_x[:,:,0,110]).reshape(-1,1)
data_y = np.squeeze(data_y[:,110]).reshape(-1)

win_size = 10
svr_predict = []
for idx in tqdm(range(0, len(data_y), win_size)):
    model = svm.SVR(gamma='auto')
    model.fit(data_x[idx:idx+win_size],data_y[idx:idx+win_size])
    svr_predict.extend(model.predict(data_x[idx:idx+win_size]))

math_utils.evaluation(svr_predict, data_y)

result = np.array(svr_predict)
np.save('model_result/svr.npy',result)

import matplotlib.pyplot as plt
plt.switch_backend('agg')

plt.title('Local_Prediction', bbox=dict(facecolor='gray', edgecolor='black', alpha=0.65))
plt.xlabel('Time slot(15-minute intervals)')
plt.ylabel('Volume(vehicle/period)')
plt.grid(linestyle='-.')

plt.plot(data_y, '-', color='blue', label="Traffic flow")
plt.plot(svr_predict, '--', color='red', label="SVR")

plt.legend(loc='upper right')
plt.savefig("./fig/"+'svr')
plt.close() 