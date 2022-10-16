import torch
import torch.nn as nn

class NADE(nn.Module):
    # weight-tying
    def __init__(self, h_size:int, input_channel:int, input_height:int, input_width:int) -> None:
        super(NADE, self).__init__()
        self.h_size = h_size
        self.input_channel = input_channel
        self.input_height = input_height
        self.input_width = input_width

        self.params = nn.ParameterDict({
            "V": nn.Parameter(torch.randn(input_channel*input_height*input_width, h_size)), 
            "b": nn.Parameter(torch.zeros(input_channel*input_height*input_width)), 
            "W": nn.Parameter(torch.randn(h_size, input_channel*input_height*input_width)), 
            "c": nn.Parameter(torch.zeros(h_size)), 
        })
        nn.init.xavier_normal_(self.params["V"])
        nn.init.xavier_normal_(self.params["W"])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: torch.Tensor((batch_size, input_channel, input_height, input_width)D), [0.0, 1.0]
        # output: torch.Tensor((batch_size, input_channel, input_height, input_width)D), [0.0, 1.0]
        x = x.view(-1, self.input_channel*self.input_height*self.input_width)
        p_hat = []
        for d in range(self.input_channel*self.input_height*self.input_width):
            # torch.Tensor((batch_size, h_size)D)
            if d == 0:
                a = (self.params["c"])[None, ...].expand(x.shape[0], -1)
            else:
                a = torch.matmul(x[:, d-1:d], self.params["W"][:, d-1:d].t()) + a
            # torch.Tensor((batch_size, h_size)D)
            h = torch.sigmoid(a)
            # torch.Tensor((batch_size, 1)D)
            p = torch.sigmoid(torch.matmul(h, self.params["V"][d]) + self.params["b"][d])[..., None]
            p_hat.append(p)
        p_hat = torch.cat(p_hat, dim=1).view(-1, self.input_channel, self.input_height, self.input_width)
        return p_hat
