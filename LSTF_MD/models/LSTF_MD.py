from copyreg import add_extension
from turtle import forward
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)  

    def forward(self, x):
    
       front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)    
       
       end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)     
       
       x = torch.cat([front, x, end], dim=1)      
       x = self.avg(x.permute(0, 2, 1))     
     
       x = x.permute(0, 2, 1)    
       return x


class Self_Atten(nn.Module):

    def __init__(self):
        super(Self_Atten,self).__init__

    def forward(self,x):
        input = x  
        sa = ScaledDotProductAttention(d_model=96, d_k=96, d_v=96, h=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output=sa(input,input,input) 
        output = output.to(device)
        return output

class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        #x = self.SelfAttention(x)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean,x
        

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__() 
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self,x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        x = x.cpu()
        x = self.dropout(x)
        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        out = out.to(device)
        return out


class Feed_Forward(nn.Module):

    def __init__(self,input_dim,hidden_dim):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,input_dim)
        
 
 
    def forward(self,x):
        output = nn.ReLU()(self.L1(x))
        output = self.L2(output)
        return output



class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(NeuralNet, self).__init__()
    
        self.layer1 = nn.Linear(in_features, hidden_features)
        
        self.relu = nn.ReLU()
       
        self.layer2 = nn.Linear(hidden_features, out_features)
 
    def forward(self, x):
        y = self.layer1(x)
        y = self.relu(y)
        y = self.layer2(y)
        return 


class NeuralNet_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
    

        super().__init__()
        self.init = nn.Linear(in_features , hidden_features)
        self.hidden = nn.Linear(hidden_features , hidden_features)
        self.outit = nn.Linear(hidden_features , out_features)
        self.relu = nn.ReLU()

   
    def forward(self, x):
        
        output = self.init(x)
        
        output = self.hidden(self.relu(output))
        
        output = self.outit(self.relu(output))
        
        return output




      
class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.ExternalAttention = External_attention()  
        self.MUSEAttention = MUSE_attention()
        self.AddNorm = Add_Norm()  
        self.FF = Feed_Forward(96 , 96)        

        # Decompsition Kernel Size
        kernel_size = 25
        
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:   
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))   
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len])) 
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))    #WT
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            
           
    
            self.MLP_sam = NeuralNet_MLP(self.seq_len,2250,self.pred_len)
            self.MLP_random = NeuralNet_MLP(self.seq_len,2250,self.pred_len)
            
            self.MLP_sam_2 = NeuralNet_MLP(self.pred_len,2250,self.pred_len)
            self.MLP_random_2 = NeuralNet_MLP(self.pred_len,2250,self.pred_len)
            
            
            self.Linear_x = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Decoder = nn.Linear(self.pred_len,self.pred_len)
          
            self.relu = nn.ReLU()
            

            
            

    def forward(self, x):

        sma_init,random_init,x_init = self.decompsition(x)
    
        sma_init, random_init,x_init = sma_init.permute(0,2,1),random_init.permute(0,2,1),x_init.permute(0,2,1) 
        #(32,96,321)->(32,321,96)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            sma_init = sma_init.to(torch.float32) 
            print('1',sma_init.shape)
            sma_output = self.MLP_sam(sma_init)   
            

            random_init = random_init.to(torch.float32)
            random_output = self.MLP_random(random_init)  
            x_output = self.Linear_x(x_init)    

            s,r,_ = self.decompsition(sma_output)
            sma_out = self.MLP_sam_2(sma_output + s)   
            random_out = self.MLP_random_2(random_output + r )
            print('4',sma_out.shape)
                     
        x = sma_out + random_out + x_output   
        #print('x',x.shape)
        #print(type(x))
        #x = self.Conv1d(x)
        x = self.Linear_Decoder(x)   
          
        return x.permute(0,2,1)     
