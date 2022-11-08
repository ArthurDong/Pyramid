import os

method = 'gru'
epoch = 10 
pred_type = 'global'

for i in range(0,50):
    f = open('result.txt', 'a+')
    f.write(method+'_'+str((i+1)*epoch)+'\n')
    f.close()
    if i == 0:
        os.system(f'python3 Compare_{method}.py --pred_type {pred_type} --epoch {epoch} --unload_model')
        #os.system(f'python3 Training_GlobalPred_Cloud.py --epoch {epoch} --unload_model')
    else:
        os.system(f'python3 Compare_{method}.py --pred_type {pred_type} --epoch {epoch}')
        #os.system(f'python3 Training_GlobalPred_Cloud.py --epoch {epoch}')

