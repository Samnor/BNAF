import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CIFAR10:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        root = "./data/cifar10"
        our_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
        trn = torchvision.datasets.CIFAR10(root, train=True, transform=our_transforms, target_transform=None, download=True)
        # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        trn = next(iter(DataLoader(trn, batch_size=len(trn))))[0].numpy()
        tst = torchvision.datasets.CIFAR10(root, train=False, transform=our_transforms, target_transform=None, download=True)
        tst = next(iter(DataLoader(tst, batch_size=len(tst))))[0].numpy()
        #trn, val, tst = load_data_normalised(file)
        print(f"self.trn: {trn.shape}")
        print(f"self.tst: {tst.shape}")
        #return
        TRAIN_COUNT = 50000
        CHANNELS = 3
        PIXEL_COUNT = 32*32
        TEST_COUNT = 10000
        
        self.trn = self.Data(trn.reshape(TRAIN_COUNT, CHANNELS*PIXEL_COUNT))
        self.val = self.Data(tst.reshape(TEST_COUNT, CHANNELS*PIXEL_COUNT))
        self.tst = self.val

        self.n_dims = self.trn.x.shape[1]

asdf = CIFAR10()
