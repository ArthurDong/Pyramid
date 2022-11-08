from typing import Tuple
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import model_utils.global_params as gp

class Linear_layer(nn.Module):
    def __init__(self, G:torch.Tensor, feature_out = 1):
        super(Linear_layer,self).__init__()
        self.G = G
        for i in range(feature_out-1):
            self.G = torch.cat([self.G,G], dim = 1) 
        self.W = nn.Parameter(torch.ones_like(self.G))

    def forward(self, x_input):
        return torch.mm(x_input, self.W.mul(self.G))

class TemporalCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route]
    def __init__(self, x_shape:list, Kt:int, c_out:int):
        super(TemporalCov,self).__init__()
        self.Kt = Kt
        self.c_out = c_out

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_route = x_shape[3]
        #layers
        self.conv_layer = nn.Conv2d(self.c_in, 2*self.c_out, (Kt,1), stride=(1,1), bias=True)
        self.actv_layer = nn.Sigmoid()

        if self.c_in > self.c_out:
            self.bias_layer = nn.Conv2d(self.c_in, self.c_out, (1,1), stride=1, bias=False)
        elif self.c_in <= self.c_out:
            self.bias_layer = nn.Parameter(torch.zeros(size=(self.batch_size, self.c_out-self.c_in, self.T_slot, self.n_route)))

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        if isinstance(self.bias_layer, nn.Conv2d):
            bias = self.bias_layer(x_input)
        elif isinstance(self.bias_layer, nn.Parameter):
            bias = torch.cat([x_input, self.bias_layer], dim=1)

        bias = bias[:, :, self.Kt-1:self.T_slot, :]
        
        conv_out = self.conv_layer(x_input)
        conv_out_a, conv_out_b = conv_out.chunk(2, dim=1)
        return (conv_out_a + bias) * self.actv_layer(conv_out_b)

class SpatioCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route]
    def __init__(self, x_shape:list, Ks:int, c_out:int):
        super(SpatioCov,self).__init__()
        self.Ks = Ks
        self.c_out = c_out

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        
        #import weight matrix
        global Kernel
        self.kernel = gp.Kernel
        #layers
        self.linear_layer = nn.Linear(Ks*self.c_in, self.c_out, bias=True)
        self.actv_layer = nn.ReLU()

        if self.c_in > self.c_out:
            self.bias_layer = nn.Conv2d(self.c_in, self.c_out, (1,1), stride=1, bias=False)
        elif self.c_in <= self.c_out:
            self.bias_layer = nn.Parameter(torch.zeros(size=(self.batch_size, self.c_out-self.c_in, self.T_slot, self.n_node)))

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        if isinstance(self.bias_layer, nn.Conv2d):
            bias = self.bias_layer(x_input)
        elif isinstance(self.bias_layer, nn.Parameter):
            bias = torch.cat([x_input, self.bias_layer], dim=1)

        #X -> [batch_size*T_slot*c_in, n_route]
        x_input = torch.reshape(x_input,(-1, self.n_node))
        #X -> [batch_size*T_slot*c_in, Ks*n_route] -> [batch_size*T_slot, c_in, Ks, n_route] -> [batch_size*T_slot, n_route, c_in, Ks] -> [batch_size*T_slot*n_route, c_in* Ks]
        x_input = torch.reshape(torch.reshape(torch.matmul(x_input, self.kernel), (-1, self.c_in, self.Ks, self.n_node)).permute(0,3,1,2) ,(-1, self.c_in*self.Ks))
        #X => [batch_size*T_slot*n_route, c_out]
        x_input = self.linear_layer(x_input)
        #X -> [batch_size, T_slot, n_route, c_out] -> [batch_size, c_out, T_slot, n_route]
        x_input = torch.reshape(x_input, (-1, self.T_slot, self.n_node, self.c_out)).permute(0,3,1,2)

        return self.actv_layer(x_input) + bias

class OutputLayer(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route]
    def __init__(self, x_shape:list, mode = 'local', classes=10):
        super(OutputLayer,self).__init__()
        self.batch_size = x_shape[0]
        self.in_feature = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        self.classes = classes
        if mode == 'local':
            self.mode = 0
            self.fc = nn.Conv2d(x_shape[1], 1, kernel_size=[x_shape[2],1], stride=[1,1])
            self.Sigmoid = nn.Sigmoid()
        elif mode == 'global':
            self.mode = 1
            #self.fc = nn.Linear(x_shape[1] * x_shape[2], classes)

            self.fc1 = nn.Linear(x_shape[1] * x_shape[2], 64)
            self.actv1 = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(64)

            self.fc2 = nn.Linear(64, 32)
            self.actv2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(32)
            
            self.fc3 = nn.Linear(32, classes)
            self.actv3 = nn.ReLU()
            self.Softmax = nn.Softmax(dim=2)

    def forward(self, x_input:torch.Tensor):
        if self.mode==0:
            x_input = self.fc(x_input).squeeze()
            return self.Sigmoid(x_input)

        elif self.mode==1:
            x_input = x_input.permute(0,3,1,2)
            x_input = x_input.reshape(self.batch_size * self.n_node, self.in_feature * self.T_slot)

            #x_input = self.fc(x_input)
            
            x_input = self.fc1(x_input)
            x_input = self.actv1(x_input)
            x_input = self.bn1(x_input)

            x_input = self.fc2(x_input)
            x_input = self.actv2(x_input)
            x_input = self.bn2(x_input)

            x_input = self.fc3(x_input)
            x_input = self.actv3(x_input)
            x_input = x_input.reshape(self.batch_size, self.n_node, self.classes)
            return self.Softmax(x_input)

class TransformCov(nn.Module):
    def __init__(self, x_shape:list, Kr:int, n_route_out:int):
        super(TransformCov,self).__init__()
        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_route = x_shape[3]
        self.n_route_out = n_route_out

        #import transform matrix
        Metrix = gp.Metrix

        Metrix2 = torch.eye(self.n_route_out)
        for i in range(Kr-1):
            Metrix2 = torch.cat([Metrix2,torch.eye(self.n_route_out)],dim=0)

        self.linear1 = Linear_layer(Metrix, feature_out=Kr)
        self.linear2 = Linear_layer(Metrix2.cuda())

        self.bn= torch.nn.BatchNorm1d(self.n_route_out*Kr).cuda()

    def forward(self, x_input):
        x_input = torch.reshape(x_input,(-1, self.n_route))

        exct_out = self.linear1(x_input) 
        exct_out = self.bn(exct_out)
        exct_out = self.linear2(exct_out) 
        exct_out = torch.reshape(exct_out, (self.batch_size, self.c_in, self.T_slot, self.n_route_out))
        return exct_out

class Pyramid(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route] 
    def __init__(self, shape_in:list, shape_out:list, feature_size_list:list, Kt:int, Ks:int, Kr:int, mode='local'):
        super(Pyramid,self).__init__()

        if mode == 'local':
            classes = 1
        elif mode == 'global':
            classes = 10

        #Add Transform Layer

        if shape_in[3] == shape_out[3]:
            pass
        else:
            self.extract = TransformCov(shape_in, Kr, shape_out[3])
            shape_in[3] = shape_out[3]

        layers = []
        for i, feature_size in enumerate(feature_size_list):
            if i%2 != 1:
                layer = TemporalCov(shape_in, Kt, feature_size)
                shape_in[2] = shape_in[2]-Kt+1
            else:
                layer = SpatioCov(shape_in, Ks, feature_size)
            shape_in[1] = feature_size
            layers.append(layer)

        self.layers = nn.Sequential(*layers)
        self.output = OutputLayer(shape_in, mode=mode, classes=classes)
    
    def forward(self, x_input:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        try:
            x_input = self.extract(x_input)
        except:
            pass
        finally:
            x_input = self.layers(x_input)
            x_output = self.output(x_input)
        return x_output, x_input


class AqGLN(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route] 
    def __init__(self, x_shape:list, Kt:int, Ks:int, Kr:int, c_out:int, n_route_out:int):
        super(AqGLN,self).__init__()
        if x_shape[3] != n_route_out:
            mode, classes = 'globalPred', 10
            self.extract = TransformCov(x_shape, Kr, n_route_out)
            x_shape[3] = n_route_out
        else:
            mode, classes = 'localPred', 10

        layers = []
        for i, out in enumerate(c_out):
            if i%2 == 1:
                layer = TemporalCov(x_shape, Kt, out)
                x_shape[2] = layer.T_slot-Kt+1
            else:
                layer = SpatioCov(x_shape, Ks, out)
            x_shape[1] = layer.c_out           
            layers.append(layer)

        self.layers = nn.Sequential(*layers)
        self.output = OutputLayer(x_shape, mode=mode, classes=classes)
    
    def forward(self, x_input:torch.Tensor):
        try:
            x_input = self.extract(x_input)
        except:
            pass
        finally:
            x_input = self.layers(x_input)
            x_output = self.output(x_input)
        return x_output, x_input

#----------------------DDNN----------------------------------
""" 
class STDCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_route]
    def __init__(self, x_shape:list, Kernel_size:int, c_out:int, UseConv = False):
        super(STDCov,self).__init__()

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_route = x_shape[3]
        #layers
        self.conv_layer = nn.Conv2d(self.c_in, c_out, (Kernel_size, 1), stride=(1,1), bias=True)
        self.linear_layer = nn.Linear(self.n_route, self.n_route, bias=True)
        self.actv_layer = nn.Sigmoid()
        if UseConv:
            self.bias_layer = nn.Conv2d(self.c_in, self.c_in, (1,1), stride=1, bias=False)
        else:
            self.bias_layer = None

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        if isinstance(self.bias_layer, nn.Conv2d):
            bias = self.bias_layer(x_input)
        else:
            bias = x_input

        #X -> [batch_size*c_in*T_slot, n_route] 
        reshape_out = torch.reshape(x_input,(-1, self.n_route))
        #SpatioFullConnection
        linear_out = self.linear_layer(reshape_out) 
        #X -> [batch_size, T_slot, c_in, n_route] -> [batch_size, c_in, T_slot, n_route]
        reshape_out = torch.reshape(linear_out,(-1, self.c_in, self.T_slot, self.n_route)) 

        #TemporalConnection
        conv_out = self.conv_layer(reshape_out + bias)

        return self.actv_layer(conv_out)

class DCNN(nn.Module): 
    #x_input [batch_size, c_in, T_slot, n_route]
    def __init__(self, x_shape:list, Kernel_size:int, c_out):
        super(DCNN,self).__init__()
        layers = []
        for i, out in enumerate(c_out):
            layer = STDCov(x_shape, Kernel_size, out)
            x_shape[2] = layer.T_slot - Kernel_size + 1
            x_shape[1] = out           
            layers.append(layer)
        
        self.layers = nn.Sequential(*layers)
        self.output = OutputLayer(x_shape)
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.layers(x_input)
        x_output = self.output(x_input)
        return x_output, x_input
"""