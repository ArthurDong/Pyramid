import torch
import torch.nn as nn
import model_utils.global_params as gp

class TemporalCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kt:int, c_out:int):
        super(TemporalCov,self).__init__()
        self.Kt = Kt
        self.c_out = c_out
        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        #layers
        self.conv_layer = nn.Conv2d(self.c_in, 2*self.c_out, (Kt,1), stride=(1,1), bias=True)
        self.actv_layer = nn.Sigmoid()

        if self.c_in > self.c_out:
            self.bias_layer = nn.Conv2d(self.c_in, self.c_out, (1,1), stride=1, bias=False)
        elif self.c_in <= self.c_out:
            self.bias_layer = nn.Parameter(torch.zeros(size=(self.batch_size, self.c_out-self.c_in, self.T_slot, self.n_node)))

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
    #x_input [batch_size, c_in, T_slot, n_node]
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

        #X -> [batch_size*T_slot*c_in, n_node]
        x_input = torch.reshape(x_input,(-1, self.n_node))
        #X -> [batch_size*T_slot*c_in, Ks*n_node] -> [batch_size*T_slot, c_in, Ks, n_node] -> [batch_size*T_slot, n_node, c_in, Ks] -> [batch_size*T_slot*n_node, c_in* Ks]
        x_input = torch.reshape(torch.reshape(torch.matmul(x_input, self.kernel), (-1, self.c_in, self.Ks, self.n_node)).permute(0,3,1,2) ,(-1, self.c_in*self.Ks))
        #X => [batch_size*T_slot*n_node, c_out]
        x_input = self.linear_layer(x_input)
        #X -> [batch_size, T_slot, n_node, c_out] -> [batch_size, c_out, T_slot, n_node]
        x_input = torch.reshape(x_input, (-1, self.T_slot, self.n_node, self.c_out)).permute(0,3,1,2)

        return self.actv_layer(x_input) + bias

class OutputLayer(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list):
        super(OutputLayer,self).__init__()
        self.fc = nn.Conv2d(x_shape[1], 1, kernel_size=[x_shape[2],1], stride=[1,1])
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.fc(x_input).squeeze()
        logits = self.Sigmoid(x_input)
        return logits

class Linear_layer(nn.Module):
    def __init__(self, G:torch.Tensor, feature_out = 1):
        super(Linear_layer,self).__init__()
        self.G = G
        for i in range(feature_out-1):
            self.G = torch.cat([self.G,G], dim = 1) 
        self.W = nn.Parameter(torch.ones_like(self.G))

    def forward(self, x_input):
        return torch.mm(x_input, self.W.mul(self.G))

class TransformCov(nn.Module):
    def __init__(self, x_shape:list, K_r:int, c_out:int):
        super(TransformCov,self).__init__()
        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        self.n_node_out = c_out

        #import transform matrix
        Metrix = gp.Metrix.cuda()
        
        Metrix2 = []
        for i in range(K_r):
            Metrix2.append(torch.eye(self.n_node_out))

        Metrix2 = torch.cat(Metrix2).cuda()

        self.linear1 = Linear_layer(Metrix, feature_out=K_r)
        self.linear2 = Linear_layer(Metrix2)

        self.bn= torch.nn.BatchNorm1d(self.n_node_out*K_r)

    def forward(self, x_input):
        x_input = torch.reshape(x_input,(-1, self.n_node))

        exct_out = self.linear1(x_input) 
        exct_out = self.bn(exct_out)
        exct_out = self.linear2(exct_out) 

        return torch.reshape(exct_out, (self.batch_size, self.c_in, self.T_slot, self.n_node_out))

class STGCN(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kt:int, Ks:int, c_out):
        super(STGCN,self).__init__()
        layers = []
        for i, out in enumerate(c_out):
            if i%2 != 1:
                layer = TemporalCov(x_shape, Kt, out)
                x_shape[2] = layer.T_slot-Kt+1
            else:
                layer = SpatioCov(x_shape, Ks, out)
            x_shape[1] = layer.c_out           
            layers.append(layer)
        
        self.layers = nn.Sequential(*layers)
        self.output = OutputLayer(x_shape)
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.layers(x_input)
        x_output = self.output(x_input)
        return x_output, x_input

class STDCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kernel_size:int, c_out:int, UseConv = False):
        super(STDCov,self).__init__()

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        #layers
        self.conv_layer = nn.Conv2d(self.c_in, c_out, (Kernel_size, 1), stride=(1,1), bias=True)
        self.linear_layer = nn.Linear(self.n_node, self.n_node, bias=True)
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

        #X -> [batch_size*c_in*T_slot, n_node] 
        reshape_out = torch.reshape(x_input,(-1, self.n_node))
        #SpatioFullConnection
        linear_out = self.linear_layer(reshape_out) 
        #X -> [batch_size, T_slot, c_in, n_node] -> [batch_size, c_in, T_slot, n_node]
        reshape_out = torch.reshape(linear_out,(-1, self.c_in, self.T_slot, self.n_node)) 

        #TemporalConnection
        conv_out = self.conv_layer(reshape_out + bias)

        return self.actv_layer(conv_out)

class DCNN(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
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

class ATTCov(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, K:int, c_out:int):
        super(ATTCov,self).__init__()
        self.K = K
        self.c_out = c_out

        self.batch_size = x_shape[0]
        self.c_in = x_shape[1]
        self.T_slot = x_shape[2]
        self.n_node = x_shape[3]
        #layers
        self.conv_layer1 = nn.Conv2d(self.T_slot, 1, (K,1), stride=(1,1), bias=True)
        self.conv_layer2 = nn.Conv2d(self.c_in, 1, (K,1), stride=(1,1), bias=True)

        self.weight = nn.Parameter(torch.rand(self.c_in-2,self.T_slot-2), requires_grad=True)
        #self.bias=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        self.actv_layer = nn.Sigmoid()

    def forward(self, x_input:torch.Tensor) -> torch.Tensor:
        conv_out_a = self.conv_layer1(x_input.permute(0,2,1,3)).squeeze(dim=1) # batch_size, c_in, n_node
        conv_out_b = self.conv_layer2(x_input.permute(0,1,2,3)).squeeze(dim=1) # batch_size, T_slot, n_node
        conv_out_a = conv_out_a.permute(0,2,1) # batch_size, n_node, c_in
        conv_out_b = conv_out_b.permute(0,2,1) # batch_size, n_node, T_slot
        x_output = self.actv_layer(torch.matmul(conv_out_a, self.weight) * conv_out_b) # batch_size, n_node, fs
        return x_output.permute(0,2,1).unsqueeze(1) # batch_size, 1,fs, n_node

class HGCN_block(nn.Module):
    #x_input [batch_size, c_in, T_slot, n_node]
    def __init__(self, x_shape:list, Kt:int, Ks:int, K:int, c_out:list):
        super(HGCN_block,self).__init__()
        self.layer0 = TemporalCov(x_shape , Kt, c_out[0])
        x_shape[2] -= Kt-1
        x_shape[1] = c_out[0]
        self.layer1 = SpatioCov(x_shape , Ks, c_out[1])
        x_shape[1] = c_out[1]
        self.layer2 = ATTCov(x_shape , K, c_out[2])
        x_shape[2] -= Kt-1
        x_shape[1] = 1
    
    def forward(self, x_input:torch.Tensor):
        x_input = self.layer0(x_input)
        x_input = self.layer1(x_input)
        x_output = self.layer2(x_input)
        return x_output

class FC_3(nn.Module):
    #x_input [batch_size, 1, fs, n_node]
    def __init__(self, x_shape:list, c_out = [8,4,1]):
        super(FC_3,self).__init__()
        self.batch_size = x_shape[0]
        self.fs = x_shape[2]
        self.n_node =  x_shape[3]

        self.fc1 = nn.Linear(self.fs, c_out[0])
        self.fc2 = nn.Linear(c_out[0],c_out[1])
        self.fc3 = nn.Linear(c_out[1],c_out[2])

        self.actv1 = nn.ReLU()
        self.actv2 = nn.ReLU()

    def forward(self, x_input:torch.Tensor):
        x_input = x_input.permute(0,3,1,2).squeeze() #x_input [batch_size, n_node, fs]
        x_input = x_input.reshape(-1,self.fs) #x_input [batch_size*n_node, fs]
        x_input = self.fc1(x_input) 
        x_input = self.actv1(x_input)
        x_input = self.fc2(x_input)
        x_input = self.actv2(x_input)
        x_input = self.fc3(x_input) #x_input [batch_size*n_node, 1]
        return x_input.squeeze().reshape(self.batch_size,self.n_node) #x_input [batch_size, n_node]