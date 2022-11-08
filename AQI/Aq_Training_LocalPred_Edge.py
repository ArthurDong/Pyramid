import model_utils.servers as sv
import model_utils.data_utils as data_utils
import model_utils.global_params as gp
import model_utils.math_utils as math_utils

#train
n_weeks = 30
test_start = 168 * (n_weeks)
epoch = 00

n_his = 12

gp.__init()

#import test instances
test_X, test_Y = data_utils.data_gen('datasets/BeijingAq.csv',
                                        loop=True,
                                        day_slot = 24, n_his = n_his,
                                        n_days = 7,
                                        index_col ='utc_time', header=0)

District = ['D1','D2','D3','D4','D5']

for Dis in District:
    #Cloud Server
    AQ_Edge = sv.AqEdgeServer(name="AQI_"+Dis , Region=Dis, n_weeks=n_weeks, c_out=[32,16,64,8,16], n_his=n_his)
    #AQ_Edge.load_model()
    AQ_Edge.train(epochs=epoch)
    AQ_Edge.save_model()

    idx = data_utils.station_gen_Aq(key=Dis)

    d_test_Y = AQ_Edge.eval(test_X[:,:,:,idx])[0]
    dtest_Y = test_Y[:,idx]
    
    static = math_utils.evaluation(dtest_Y,d_test_Y)

    for stat in static.keys():
        res = ''+str(stat)+':'+str(static[stat])
        print(res)

#import random
#from model_utils import plot

#seed = random.randint(0,_evalBeijing.shape[1]-1)
#plot.plot_AQI((_evalBeijing[:,seed],evalBeijing[:,seed]),"fig/"+'station_'+str(seed))"""