import numpy as np
import torch 
from scipy import io
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split

from torchvision.datasets import MNIST

def de_onehot(y_onehot):
    return np.array(list(map(lambda x: np.argmax(x), y_onehot)))

class MNISTDataset(Dataset):
    
    def __init__(self, data_dir="./data", split="train", device=torch.device("cuda")):

        assert split in ["train", "test"]
        self.split = split
        
        if split == "train":
            mnist_train_dataset = MNIST(data_dir, download=True, train=True)
     
            train_X, train_y = [], []
            for x, y in mnist_train_dataset:
                x = np.array(x)[np.newaxis,:,:]
                train_X.append(x)
                train_y.append(y)
            train_X = np.stack(train_X)
            train_y = np.stack(train_y)
            self.train_data = (train_X, train_y)
        else:

            mnist_test_dataset = MNIST(data_dir, train=False)
        
            test_X, test_y = [], []
            for x, y in mnist_test_dataset:
                x = np.array(x)[np.newaxis,:,:]
                test_X.append(x)
                test_y.append(y)
            test_X = np.stack(test_X)
            test_y = np.stack(test_y)
            self.test_data = (test_X, test_y)

        if split == "train":
            self.data = TensorDataset(torch.Tensor(train_X).to(device), torch.Tensor(train_y).to(device))
        else:
            self.data = TensorDataset(torch.Tensor(test_X).to(device), torch.Tensor(test_y).to(device))

    def __len__(self):
        
        if self.split == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)
      
    def __getitem__(self, idx):
        
        x, y = self.data[idx]
        return x/255.0, y

def load_data(data_path, test_size, seed, default=True):
    
    if default:

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
    
    else:

        train_dataset = MNISTDataset("./data", split="train")
        test_dataset = MNISTDataset("./data", split="test")

        return train_dataset, test_dataset

def create_dataloader(X, y, batch_size, seed, shuffle=True):
    torch.manual_seed(seed)
    td = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    return DataLoader(td, batch_size=batch_size, shuffle=shuffle)
