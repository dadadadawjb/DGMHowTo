import torch
import torch.nn as nn

class Generator(nn.Module):
    # MLP
    def __init__(self, z_size:int, output_channel:int, output_height:int, output_width:int) -> None:
        super(Generator, self).__init__()
        self.z_size = z_size
        self.output_channel = output_channel
        self.output_height = output_height
        self.output_width = output_width

        self.layers = nn.Sequential(
            nn.Linear(z_size, 2*z_size), 
            nn.BatchNorm1d(2*z_size), 
            nn.LeakyReLU(0.2), 

            nn.Linear(2*z_size, 4*z_size), 
            nn.BatchNorm1d(4*z_size), 
            nn.LeakyReLU(0.2), 

            nn.Linear(4*z_size, 8*z_size), 
            nn.BatchNorm1d(8*z_size), 
            nn.LeakyReLU(0.2), 

            nn.Linear(8*z_size, output_channel * output_height * output_width), 
            nn.Tanh(), 
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor((batch_size, z_size)D), N(0.0, 1.0)
        # output: torch.Tensor((batch_size, output_channel, output_height, output_width)D), [-1.0, 1.0]
        return self.layers(x).view(-1, self.output_channel, self.output_height, self.output_width)


class Discriminator(nn.Module):
    # MLP
    def __init__(self, z_size:int, input_channel:int, input_height:int, input_width:int) -> None:
        super(Discriminator, self).__init__()
        self.z_size = z_size                    # actually no need expose `z_size` to `Discriminator`, for symmetry
        self.input_channel = input_channel
        self.input_height = input_height
        self.input_width = input_width

        # aim to make the discriminator weaker than the generator
        self.layers = nn.Sequential(
            nn.Linear(input_channel * input_height * input_width, 8*z_size), 
            nn.BatchNorm1d(8*z_size), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3), 

            # nn.Linear(8*z_size, 4*z_size), 
            # nn.BatchNorm1d(4*z_size), 
            # nn.LeakyReLU(0.2), 
            # nn.Dropout(0.3), 

            # nn.Linear(4*z_size, 2*z_size), 
            # nn.BatchNorm1d(2*z_size), 
            # nn.LeakyReLU(0.2), 
            # nn.Dropout(0.3), 

            nn.Linear(8*z_size, 2*z_size), 
            nn.BatchNorm1d(2*z_size), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3), 

            nn.Linear(2*z_size, 1), 
            nn.Sigmoid(), 
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor((batch_size, input_channel, input_height, input_width)D), N(0.0, 1.0) or [-1.0, 1.0]
        # output: torch.Tensor((batch_size, 1)D), [0.0, 1.0]
        return self.layers(x.view(-1, self.input_channel * self.input_height * self.input_width))
