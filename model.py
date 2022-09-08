import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3)
        self.fc1 = nn.Linear(4096, 1600)

    def forward(self, x):
        
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        #flatten => x shape = (batch size, flatten features)
        
        x = self.fc1(x)
        x = F.dropout(x,training=self.training) #dropout 
        
        return x
    
    def num_flat_features(sef, x): #flatten function
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=160, hidden_size=256, num_layers = 2,batch_first = True)
        self.fc1 = nn.Linear(2560,1)
       
    def forward(self,x):
        #LSTM input shape => (batch,seq_len,features)
        x = x.reshape(-1,10,160)
        x, states = self.lstm1(x)
        x = x.reshape(-1, 2560)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    
class C_LSTM(nn.Module):
    def __init__(self):
        super(C_LSTM,self).__init__()
        self.lenet = LeNet()
        self.lstm = LSTM()
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.lenet(x)
        x = self.lstm(x)
        return x