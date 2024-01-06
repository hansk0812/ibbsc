import numpy as np
import torch 
from scipy import io
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset

def de_onehot(y_onehot):
    return np.array(list(map(lambda x: np.argmax(x), y_onehot)))


def load_data(data_path, test_size, seed):
    # Load data as is
    data = io.loadmat(data_path) # OBS loads in a weird JSON
    X = data["F"] # (4096, 12)
    y = data["y"] # (1, 4096)
    
    y = y.squeeze() # (4096, )
    # Uncomment below if onehot is needed.
    #classes = len(np.unique(y))
    #y_onehot = np.eye(classes)[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=seed,
                                                        test_size=test_size, 
                                                        shuffle=True,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def load_data_mnist(data_path, test_size, seed):

    train_dset = MNIST('./mnist', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_dset = MNIST('./mnist', train=False,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]))

    X_dset, y_dset = [], []
    for x, y in train_dset:
        x = x.reshape((28*28))
        X_dset.append(x)
        y_dset.append(y)
    for x, y in test_dset:
        x = x.reshape((28*28))
        X_dset.append(x)
        y_dset.append(y)
    
    X = torch.stack(X_dset, axis=0)
    y = torch.Tensor(y_dset)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=seed,
                                                        test_size=test_size, 
                                                        shuffle=True,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def create_dataloader(X, y, batch_size, seed, shuffle=True):
    torch.manual_seed(seed)
    td = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    return DataLoader(td, batch_size=batch_size, shuffle=shuffle)
