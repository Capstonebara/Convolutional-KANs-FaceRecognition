from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class KANC_MLP(nn.Module):
    def __init__(self, grid_size: int = 10):
        super().__init__()
        # Modified input channels for RGB
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3,  # Changed from 1 to 3 for RGB
            out_channels=5,
            kernel_size=(3,3),
            grid_size=grid_size
        )

        self.conv2 = KAN_Convolutional_Layer(
            in_channels=5,
            out_channels=5,
            kernel_size=(3,3),
            grid_size=grid_size
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()
        
        # Calculate new flattened size for 112x112 input:
        # Input: 112x112x3
        # After conv1: 110x110x5
        # After pool1: 55x55x5
        # After conv2: 53x53x5
        # After pool1: 26x26x5 = 3380 features
        
        # Changed output to 512
        self.linear1 = nn.Linear(5 * 26 * 26, 512)
        self.name = f"KANC MLP (Small) (gs = {grid_size})"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x) 
        x = self.conv2(x)
        x = self.pool1(x)

        print("Shape before flattening:", x.shape)
        x = self.flat(x)
        x = self.linear1(x)
        return x 
    

class KANC_MLP_Medium(nn.Module):
    def __init__(self,grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3,
            out_channels= 5,
            kernel_size= (3,3),
            grid_size = grid_size
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 10,
            kernel_size = (3,3),
            grid_size = grid_size
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(5 * 26 * 26, 512)
        self.name = f"KANC MLP (Medium) (gs = {grid_size})"


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x) 
        x = self.conv2(x)
        x = self.pool1(x)

        print("Shape before flattening:", x.shape)
        x = self.flat(x)
        x = self.linear1(x)
        return x 




