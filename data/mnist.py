import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNIST:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        root = "./mnist"
        trn = torchvision.datasets.MNIST(root, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]), target_transform=None, download=True)
        # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        trn = next(iter(DataLoader(trn, batch_size=len(trn))))[0].numpy()
        tst = torchvision.datasets.MNIST(root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]), target_transform=None, download=True)
        tst = next(iter(DataLoader(tst, batch_size=len(tst))))[0].numpy()
        #trn, val, tst = load_data_normalised(file)
        TRAIN_COUNT = 60000
        PIXEL_COUNT = 28*28
        TEST_COUNT = 10000
        
        self.trn = self.Data(trn.reshape(TRAIN_COUNT, PIXEL_COUNT))
        self.val = self.Data(tst.reshape(TEST_COUNT, PIXEL_COUNT))
        self.tst = self.val
        #print(f"self.trn: {self.trn.x.shape}")
        #print(f"self.tst: {self.tst.x.shape}")
        self.n_dims = self.trn.x.shape[1]