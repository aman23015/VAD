import torch 
import argparse
import torch.nn as nn
from torchinfo import summary


class Depth_wise(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',
                *args, **kwargs) -> None:
        super().__init__()
        
        self.Convolution = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode),
            nn.Conv1d(in_channels, out_channels, 1, stride, padding, dilation, groups, bias, padding_mode)
        )
    def forward(self,x):
        x = self.Convolution(x)
        return x
    
class Sub_Block(nn.Module):
    def __init__(self,
                 in_channels:int,
                 C : int ,
                 kernel_size : int,
                 dropout : float = 0.1,
                 *args,**kwargs):
        super().__init__()
        self.dpc1 = Depth_wise(in_channels,C,kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(C)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout1d(p = dropout)
    
    def forward(self,x):
        return self.dropout(self.relu1(self.bn1(self.dpc1(x))))

class Block(nn.Module):
    def __init__(self,
                 R:int,
                 in_channels:int ,
                 out_channels : int,
                 kernel_size : int ,
                 dropout:float = 0.1,
                 *args,**kwargs
                 )->None:
        super().__init__()
        self.sub_blocks = R
        self.in_channels = in_channels
        self.Block_list =nn.ModuleList()
        for _ in range(self.sub_blocks-1):
            sub_block = Sub_Block(in_channels,out_channels,kernel_size,dropout)
            self.in_channels=out_channels
            self.Block_list.append(sub_block)
        # self.subblock = Sub_Block(in_channels,out_channels,kernel_size=kernel_size,dropout=dropout)
        self.Pointwise_Conv_Residual = nn.Conv1d(in_channels,out_channels,1)
        self.BN_Residual = nn.BatchNorm1d(out_channels) 

        self.last_block_depthwise_operation = Depth_wise(self.in_channels,out_channels,kernel_size)
        self.BN_last_block = nn.BatchNorm1d(out_channels)
        self.relu_last_block = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout)
        self.ada_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        residual = x
        for block in self.Block_list:
            x = block(x) 
        residual = self.BN_Residual(self.Pointwise_Conv_Residual(residual))
        x = self.BN_last_block(self.last_block_depthwise_operation(x))

        self.ada_pooling = nn.AdaptiveAvgPool1d(x.shape[2:])
        residual = self.ada_pooling(residual)
        x = x + residual
        x = self.dropout(self.relu_last_block(x))
        return x    

     
        


class Marble_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 B : int = 3,
                 R : int = 2, 
                 C : int = 64,
                 kernel_size_blockwise : list = [13,15,17],
                 num_classes : int = 2,
                 dropout : float = 0.1,
                 Linear_Layer_neurons : int = 14556,
                 *args,**kwargs)-> None:
        super().__init__()
        self.B = B
        self.R = R
        self.C = C
        self.num_classes = num_classes
        self.in_channels = in_channels
        # prologue
        self.conv_prologue = nn.Conv1d(in_channels=in_channels,out_channels=128,kernel_size=11)
        self.bn_prologue = nn.BatchNorm1d(128)
        self.relu_prologue = nn.ReLU()
        self.in_channels = 128

        # Blocks
        self.Blocks = nn.ModuleList()

        if len(kernel_size_blockwise) != self.B:
            raise Exception("len(kernel_size_blockwise) != self.B")

        for kernel_size in kernel_size_blockwise:
            block = Block(self.R,self.in_channels,out_channels=self.C,kernel_size=kernel_size,dropout=dropout)
            self.in_channels = self.C # 64
            self.Blocks.append(block)
        #Epilogue
        self.conv_epilogue_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.C,out_channels=128,kernel_size=29,dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )    
        self.conv_epilogue_2 = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv_epilogue_3 = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels= num_classes,kernel_size=29,dilation=2),
            nn.BatchNorm1d(num_classes),
            nn.ReLU()
        )


        self.ada_pooling = nn.AdaptiveAvgPool1d(1)
        ### Classification Network ###
        self.pooling_layer = nn.MaxPool1d(kernel_size=7, stride=3,dilation=2)
        if Linear_Layer_neurons == None:
            raise Exception("Linear_Layer_neurons == None")

        self.pooling_layer = nn.MaxPool1d(kernel_size=7, stride=3,dilation=2)
        print(self.out_channels)
        self.Linear0 = nn.Linear(Linear_Layer_neurons, 2048)
        self.activation0 = nn.ReLU()
        self.Linear1 = nn.Linear(2048, 512)
        self.activation1 = nn.ReLU()
        self.Linear2 = nn.Linear(512,32)
        self.activation2 = nn.ReLU()
        self.Linear3 = nn.Linear(32,2)
        self.activation3 = nn.ReLU()

    def forward(self,x):
        x = self.relu_prologue(self.bn_prologue(self.conv_prologue(x)))
        x_residual = x

        for block in self.Blocks:
            x = block(x)
          
            self.ada_pooling = nn.AdaptiveAvgPool2d(x.shape[1:])            
            x_residual = self.ada_pooling(x_residual)
            x = x + x_residual
            x_residual = x
        x = self.conv_epilogue_1(x)
        x = self.conv_epilogue_2(x)
        x = self.conv_epilogue_3(x)
        x = self.pooling_layer(x)
        x = x.reshape(x.size(0),-1) 
        x = self.Linear0(x)
        x = self.activation0(x)
        x = self.Linear1(x)
        x = self.activation1(x)
        x = self.Linear2(x)
        x = self.activation2(x)
        x = self.Linear3(x)
        x = self.activation3(x)
        
        return x 


# sample rate : 44100 
#Linear_layer_neurons for 16000 sample rate : 10522
#Linear_layer_neurons for 44100 sample rate : 29256
#Linear layer neurons for 66150 sample rate : 43956
# formula to calculate this is (((sample_rate-206)+2*padding - dilation*(kernel_size -1)-1)/stride +1)*2
# device = torch.device(f"cuda:{cuda}") if cuda in range(0,torch.cuda.device_count()) else torch.device("cpu")
# one block increase = -22 in the Linear_layer_neurons. 

if __name__ == "__main__":
# # #     # CUDA_VISIBLE_DEVICES =""
    
# #     # device = torch.device("cpu")
# #     device = torch.device(f"cuda:{cuda}") if cuda in range(0,torch.cuda.device_count()) else torch.device("cpu")
#14554

    model = Marble_Net(1,6,2,64,[13,15,17,19,21,21],2,Linear_Layer_neurons=14478)
# #     model = model.to(device)

    # print(summary(model, (5,1,22050)))
    x = torch.randn(5,1,22050)
# # # #     x = x.to(device)
    print(model(x).shape)       
            
        