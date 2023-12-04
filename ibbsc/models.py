import torch 
from torch import nn
from torch.nn import functional as F
from custom_exceptions import ActivationError

from torchvision.models import vgg16_bn

# Supported activation functions for hidden layers except output
activations_functions = {
    "tanh" : torch.tanh,
    "relu" : F.relu,
    "relu6" : F.relu6,
    "elu" : F.elu,
    "linear" : "linear",
    "lrelu" : F.leaky_relu
}

class VGG(nn.Module):
    #vgg["features"][0,3,7,10,14,17,20,24,27,30,34,37,40], vgg["classifier"][0,3,6]
    #[(3,64),(64,64),(64,128),(128,128),(128,256),(256,256),(256,256),(256,512),(512,512),(512,512),(512,512),(512,512),(512,512)]
    #[(25088,4096),(4096,4096),(4096,1000)]

    def __init__(self, num_classes=10):
        super().__init__()
        self.vgg_net = vgg16_bn()
        self.final_layer = nn.Linear(4096, num_classes)

        self.layer_indices = [(2,5,9,12,16,19,22,26,29,32,36,39,42), (2,5,8)]

    def forward(self, x):
        
        features, pool, classifier = self.vgg_net.children()
        
        activations = []

        for idx, feat in enumerate(features):
            print (x.shape)
            print (feat(x).shape)
            x = feat(x) 
            if idx in self.layer_indices[0]:
                activations.append(x)
        x = pool(x)
        x = x.reshape((x.shape[0], -1))
        for idx, feat in enumerate(classifier):
            x = feat(x)
            if idx in self.layer_indices[1]:
                activations.append(x)
        x_softmax = F.softmax(x, dim=-1)

        return x, x_softmax, activations

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

