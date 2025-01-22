import torch
import argparse
import torch.nn as nn
from torchinfo import summary

P = argparse.ArgumentParser()
P.add_argument("gpu",type=int,default=0)
A = P.parse_args()

cuda = A.gpu

class Depthwise(nn.Module):
    """
    Depthwise convolution followed by pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate', or 'circular'. Defaults to 'zeros'.
    """
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
        
        # self.Convolution = nn.Sequential(
        self.conv1 =    nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode)
        self.conv2 =    nn.Conv1d(in_channels, out_channels, 1, stride, padding, dilation, groups, bias, padding_mode)
        # )
    
    def forward(self, INPUT):
        """
        Perform forward pass through the Depthwise convolution module.

        Args:
            INPUT: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = INPUT
        # print("Depth_wise")
        # print(x.shape)
        x= self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        # input("wait")
        return x
    


class SubBlock(nn.Module):
    """
    Sub-block comprising depthwise convolution, batch normalization, ReLU activation, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        padding_mode (str, optional): 'zeros', 'reflect', 'replicate', or 'circular'. Defaults to 'zeros'.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
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
                 dropout: float = 0.1
                 ) -> None:
        super(SubBlock,self).__init__()
        # print(in_channels)
        # print(out_channels)
        # print(kernel_size)
        # input("wait")
        self.DepthWise_and_pointwise_convlution = Depthwise(
            in_channels,out_channels,kernel_size,stride,
            padding,dilation,groups,bias,padding_mode
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout)  
    
    def forward(self,x):
        """
        Perform forward pass through the SubBlock module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.DepthWise_and_pointwise_convlution(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
    


class Block(nn.Module):
    """
    Block comprising multiple SubBlocks followed by residual connections and final transformations.

    Args:
        R (int): Number of repetitions of SubBlocks.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self,
                 R: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dropout: float = 0.1
                 ) -> None:
        super(Block,self).__init__()
        self.Block_list = nn.ModuleList()
        # print("self_blocks_list ",self.Block_list)

        self.sub_blocks = R
        self.in_channels = in_channels

        for _ in range(self.sub_blocks-1):
            sub_block = SubBlock(self.in_channels,out_channels,kernel_size)
            self.in_channels = out_channels
            self.Block_list.append(sub_block)   
        
        self.Pointwise_Conv_Residual = nn.Conv1d(in_channels, out_channels, 1)
        self.BN_Residual = nn.BatchNorm1d(out_channels)

        self.last_block_depthwise_operation = Depthwise(self.in_channels,out_channels,kernel_size)
        self.BN_last_block = nn.BatchNorm1d(out_channels)
        self.relu_last_block = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout)

        self.ada_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        """
        Perform forward pass through the Block module.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
    


class MarbleNet(nn.Module):
    """
    MarbleNet architecture consisting of Prologue, Blocks, and Epilogue sections, followed by a classification network.

    Args:
        B (int): Number of blocks.
        R (int): Number of repetitions of SubBlocks in each block.
        C (int): Number of output channels in each block.
        in_channels (int): Number of input channels.
        kernel_size_prologue (int): Size of the convolutional kernel in the prologue section.
        kernel_size_blockwise (list): List of kernel sizes for each block.
        kernel_size_Eplilogue (int): Size of the convolutional kernel in the epilogue section.
        number_classes (int): Number of output classes.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        Linear_Layer_neurons (int, optional): Number of neurons in the linear layer. Defaults to None.
    """
    def __init__(self,
                 B: int,
                 R: int,
                 C: int,
                 in_channels: int,
                 kernel_size_prologue: int,
                 kernel_size_blockwise: list,
                 kernel_size_Eplilogue: int,
                 number_classes: int,
                 dropout: int = 0.1,
                 Linear_Layer_neurons: int = None
                 ) -> None:
        super(MarbleNet,self).__init__()

        self.B = B
        self.R = R
        self.C = C
        self.in_channels = in_channels # 1
        ### Prologue Block ###
        
        self.Conv_Prologue = nn.Conv1d(in_channels=self.in_channels,out_channels=128,kernel_size=kernel_size_prologue)
        self.BN_Prologue = nn.BatchNorm1d(128)
        self.Relu_Prolugue = nn.ReLU()
        self.in_channels = 128 # 128

        ### B Blocks ###
        self.Blocks = nn.ModuleList()
        print("self_blocks ",self.Blocks)

        if len(kernel_size_blockwise) != self.B:
            raise Exception("len(kernel_size_blockwise) != self.B")

        for kernel_size in kernel_size_blockwise:
            block = Block(self.R,self.in_channels,out_channels=self.C,kernel_size=kernel_size,dropout=dropout)
            self.in_channels = self.C # 64
            self.Blocks.append(block)

        ### Eplilogue Dilation ###

        self.epilogue = nn.Sequential(
            nn.Conv1d(self.in_channels,128,kernel_size_Eplilogue,dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,number_classes,1)
        )
        self.in_channels = number_classes
        self.ada_pooling = nn.AdaptiveAvgPool1d(1)

        ### Classification Network ###

        if Linear_Layer_neurons == None:
            raise Exception("Linear_Layer_neurons == None")

        self.pooling_layer = nn.MaxPool1d(kernel_size=7, stride=3,dilation=2)

        self.Linear0 = nn.Linear(Linear_Layer_neurons, 2048)
        self.activation0 = nn.ReLU()
        self.Linear1 = nn.Linear(2048, 512)
        self.activation1 = nn.ReLU()
        self.Linear2 = nn.Linear(512,32)
        self.activation2 = nn.ReLU()
        self.Linear3 = nn.Linear(32,2)
        self.activation3 = nn.ReLU()

    def forward(self,x):
        x = self.Relu_Prolugue(self.BN_Prologue(self.Conv_Prologue(x)))
        x_residual = x

        for block in self.Blocks:
            x = block(x)
            self.ada_pooling = nn.AdaptiveAvgPool2d(x.shape[1:])
            x_residual = self.ada_pooling(x_residual)
            x = x + x_residual
            # print('after adding residual',x.shape)
            
            x_residual = x
        print(x.shape)
        x = self.pooling_layer(x)
        print(x.shape)

        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.Linear0(x)
        x = self.activation0(x)

        x = self.Linear1(x)
        x = self.activation1(x)

        x = self.Linear2(x)
        x = self.activation2(x)

        x = self.Linear3(x)
        x = self.activation3(x)

        return x



def Inference_For_MarbleNet(Audio, label, sample_rate, type_of_smoothing="Simple", logits=None):
    """
    Audio = Chunks of an audio, must be an list of chunks (chunks can be of 1-D tensor)
    label = regarding each chunk from Audio tell whether there is speech or no-speech
    logits = output from MarbleNet model
    type_of_smoothing = [Simple, Recursive, logits_based]
    sample_rate: sample_rate

    Assuming Audio contains chunks like:
    x - x' , x'- x'' ,  x'' - x''' , .....
    In that, while merging the two audio we have to remove 0'th index of the second chunk because it will already present in the merged sequence.
    """

    if type_of_smoothing not in ["Simple", "Recursive", "logits_based"]:
        raise Exception("smoothing should be [Simple, Recursive, logits_based]")

    if type_of_smoothing == "logits_based" and logits is None:
        raise Exception("for logits based smoothing logits should not be none")

    if len(Audio) != len(label):
        raise Exception("len(Audio) != len(label)")

    return_smoothed_out_audio_with_merged_chunks = []
    return_smoothed_out_label_with_merged_chunks = []

    if type_of_smoothing == "Simple":
        """
        Simple Merging based upon label, i.e., For a particular index 'i' if there is speech on 'i-1' merged both of them WLOG for no-speech, else don't merge them.
        """
        prev_chunk_label = label[0]
        prev_chunk_audio = Audio[0]

        for curr_chunk_audio, curr_chunk_label in zip(Audio[1:], label[1:]):
            if curr_chunk_label != prev_chunk_label:
                prev_chunk_audio = prev_chunk_audio.squeeze(dim=1)
                return_smoothed_out_audio_with_merged_chunks.append(prev_chunk_audio)
                return_smoothed_out_label_with_merged_chunks.append(prev_chunk_label)
                prev_chunk_audio = curr_chunk_audio
                prev_chunk_label = curr_chunk_label
            else:
                prev_chunk_audio = torch.cat((prev_chunk_audio, curr_chunk_audio[1:]), dim=0)
                prev_chunk_label = curr_chunk_label

        prev_chunk_audio = prev_chunk_audio.squeeze(dim=1)
        return_smoothed_out_audio_with_merged_chunks.append(prev_chunk_audio)
        return_smoothed_out_label_with_merged_chunks.append(prev_chunk_label)

    elif type_of_smoothing == "Recursive":
        """
        Based upon predecessor and successor i.e., for a particular index i check if both its label[predecessor] 
        and label[successor] are the same then make label[i] = label[successor].
        """
        smoothed_out_label = [label[0]]

        for current in range(1, len(label) - 1):
            predecessor = max(0, current - 1)
            successor = min(len(label) - 1, current + 1)
            if label[predecessor] == label[successor]:
                smoothed_out_label.append(label[successor])
            else:
                smoothed_out_label.append(label[current])

        return_smoothed_out_audio_with_merged_chunks, return_smoothed_out_label_with_merged_chunks = Inference_For_MarbleNet(Audio, smoothed_out_label, sample_rate, "Simple")

    elif type_of_smoothing == "logits_based":
        pass

    return return_smoothed_out_audio_with_merged_chunks, return_smoothed_out_label_with_merged_chunks


if __name__ == "__main__":
    device = torch.device(f"cuda:{cuda}") if cuda in range(0,torch.cuda.device_count()) else torch.device("cpu")

    model = MarbleNet(3,2,64,1,11,[13,15,17],29,2,Linear_Layer_neurons=43956)
    # model = model.to(device)
    # print(summary(model, (64,1,16000)))

    x = torch.randn(5,1,66150)
    # x= x.to(device)
    print(model(x).shape)


