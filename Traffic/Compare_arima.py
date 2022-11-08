import numpy as np
import model_utils.math_utils as math_utils
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA
from model_utils import data_utils

routes = data_utils.station_gen(key=["City of Fremont","City of Oakland","City of Richmond","City of San Francisco","City of San Jose"])

_, train_data = data_utils.data_gen(n_route=routes, n_days=3*7, n_his=1, loop=True, offset=0)
_, test_data = data_utils.data_gen(n_route=routes, n_days=7, n_his=1, loop=True, offset=3024)

arima_train, arima_test = train_data[:,110], test_data[:,110]

history = [x for x in arima_train]
predictions = []

win_size = 10
for t in tqdm(range(0, len(arima_test), win_size)):
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit(disp=-1) #,disp=-1, method='css'
    output = model_fit.forecast(win_size)
    yhat = output[0]
    predictions.extend(yhat)
    obs = arima_test[t:t+win_size]
    history.extend(obs)

predictions = np.array(predictions)[:1008]
math_utils.evaluation(arima_test, predictions)

result = np.array(arima_test)
np.save('model_result/arima.npy',result)

import matplotlib.pyplot as plt
# plot
plt.switch_backend('agg')

plt.title('Local_Prediction', bbox=dict(facecolor='gray', edgecolor='black', alpha=0.65))
plt.xlabel('Time slot(15-minute intervals)')
plt.ylabel('Volume(vehicle/period)')
plt.grid(linestyle='-.')
#plt.ylim(0,1)

plt.plot(arima_test, '-', color='blue', label="Traffic flow")
plt.plot(predictions, '--', color='red', label="ARIMA")

plt.legend(loc='upper right')

plt.savefig("./fig/"+'arima')
plt.close()  