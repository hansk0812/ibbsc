import torch 
from torch import nn
from torch.nn import functional as F
from custom_exceptions import ActivationError



# Supported activation functions for hidden layers except output
activations_functions = {
    "tanh" : torch.tanh,
    "relu" : F.relu,
    "relu6" : F.relu6,
    "elu" : F.elu,
    "linear" : "linear",
    "lrelu" : F.leaky_relu
}


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation="tanh", seed=0):
        super(FNN, self).__init__()
        torch.manual_seed(seed)
        self.activation = activation
        self.num_layers = len(layer_sizes)
        self.inp_size = layer_sizes[0]
        
        in_out_pairs = [(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.linears = nn.ModuleList([nn.Linear(*pair) for pair in in_out_pairs])
    
    def forward(self, x):
        activations = [] 
        for idx in range(self.num_layers-2):
            x = self.linears[idx](x)
            # TODO: Maybe just pass the actual function to the constructor.
            # However this also restrict it to the activation function that
            # the mutual information estimation is supported of currently.
            if not activations_functions.get(self.activation):
                raise ActivationError("Activation Function not supported...")
            if not self.activation == "linear":
                x = activations_functions[self.activation](x)
            if not self.training: #Internal flag 
                activations.append(x)
        x = self.linears[-1](x)
        x_softmax = F.softmax(x, dim=-1)
        if not self.training: 
            # Cross entropy loss in pytorch adds softmax(x) 
            activations.append(x_softmax) 
            
        return x, x_softmax, activations
