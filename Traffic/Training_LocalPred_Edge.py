import torch
import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.global_params as gp

#train
n_weeks = 3
n_days = 7
n_his = 13
epoch = 500

gp.__init()

#Edge Server1
Fremont = sv.EdgeServer(city="City of Fremont", n_weeks=n_weeks, n_his=n_his)
Fremont.load_model()
Fremont.train(epoch=epoch)
Fremont.save_model()

#Edge Server2
Oakland = sv.EdgeServer(city="City of Oakland", n_weeks=n_weeks, n_his=n_his)
Oakland.load_model()
Oakland.train(epoch=epoch)
Oakland.save_model()

#Edge Server3
Richmond = sv.EdgeServer(city="City of Richmond", n_weeks=n_weeks, n_his=n_his)
Richmond.load_model()
Richmond.train(epoch=epoch)
Richmond.save_model()

#Edge Server4
SanFrancisco = sv.EdgeServer(city="City of San Francisco", n_weeks=n_weeks, n_his=n_his)
SanFrancisco.load_model()
SanFrancisco.train(epoch=epoch)
SanFrancisco.save_model()

#Edge Server5
SanJose = sv.EdgeServer(city="City of San Jose", n_weeks=n_weeks, n_his=n_his)
SanJose.load_model()
SanJose.train(epoch=epoch)
SanJose.save_model()

#eval
import model_utils.math_utils as math_utils

#import test instances
train_data, test_data = data_utils.data_gen('datasets/PemsD4.csv', loop=True, n_days=7, offset=3024, n_his=n_his)
train_data = torch.tensor(train_data)
test_data = torch.tensor(test_data)

routes_Fremont = data_utils.station_gen(key="City of Fremont")
routes_Oakland = data_utils.station_gen(key="City of Oakland")
routes_Richmond = data_utils.station_gen(key="City of Richmond")
routes_SanFrancisco = data_utils.station_gen(key="City of San Francisco")
routes_SanJose = data_utils.station_gen(key="City of San Jose")

#Edge pred
outputFremont = Fremont.eval(train_data[:,:,:,routes_Fremont])[0].detach().cpu().numpy()
outputOakland = Oakland.eval(train_data[:,:,:,routes_Oakland])[0].detach().cpu().numpy()
outputRichmond = Richmond.eval(train_data[:,:,:,routes_Richmond])[0].detach().cpu().numpy()
outputSanFrancisco = SanFrancisco.eval(train_data[:,:,:,routes_SanFrancisco])[0].detach().cpu().numpy()
outputSanJose = SanJose.eval(train_data[:,:,:,routes_SanJose])[0].detach().cpu().numpy()

#Orignal data
_outputFremont = test_data[:,routes_Fremont].detach().cpu().numpy()
_outputOakland = test_data[:,routes_Oakland].detach().cpu().numpy()
_outputRichmond = test_data[:,routes_Richmond].detach().cpu().numpy()
_outputSanFrancisco = test_data[:,routes_SanFrancisco].detach().cpu().numpy()
_outputSanJose = test_data[:,routes_SanJose].detach().cpu().numpy()

#evaluate routes
print("----Edge Server of Fremont----")
math_utils.result_print(math_utils.evaluation(_outputFremont,outputFremont))
print("----Edge Server of Oakland----")
math_utils.result_print(math_utils.evaluation(_outputOakland,outputOakland))
print("----Edge Server of Richmond----")
math_utils.result_print(math_utils.evaluation(_outputRichmond,outputRichmond))
print("----Edge Server of San Francisco----")
math_utils.result_print(math_utils.evaluation(_outputSanFrancisco,outputSanFrancisco))
print("----Edge Server of San Jose----")
math_utils.result_print(math_utils.evaluation(_outputSanJose,outputSanJose))

