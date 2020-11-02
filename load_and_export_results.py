import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
import pprint
import datetime
import torch
from torch.utils import data
from bnaf import *
from tqdm import tqdm
from optim.adam import Adam
from optim.lr_scheduler import ReduceLROnPlateau
import scipy.stats as stats


from data.gas import GAS
from data.bsds300 import BSDS300
from data.hepmass import HEPMASS
from data.miniboone import MINIBOONE
from data.power import POWER
from data.mnist import MNIST
from data.fashionmnist import FASHIONMNIST
from data.fashionmnist_aug import FASHIONMNISTAUG
from data.mnist_aug import MNISTAUG
from data.svhn import SVHN
from data.cifar10 import CIFAR10
from data.cifar10_grayscale import CIFAR10GRAYSCALE
from data.svhn_grayscale import SVHNGRAYSCALE
from data.fashionmnist_efficient import FASHIONMNISTEFFICIENT
from data.mnist_efficient import MNISTEFFICIENT
from data.svhn_grayscale_efficient import SVHNGRAYSCALEEFFICIENT
from data.cifar10_grayscale_efficient import CIFAR10GRAYSCALEEFFICIENT

import time
import ast
import json

NAF_PARAMS = {
    'power': (414213, 828258),
    'gas': (401741, 803226),
    'hepmass': (9272743, 18544268),
    'miniboone': (7487321, 14970256),
    'bsds300': (36759591, 73510236),
    'fmnist': (1337, 1337),
    'mnist': (1337, 1337),
    'svhn': (1337, 1337),
    'cifar10': (1337, 1337),
    'svhngray': (1337, 1337),
    'cifar10gray': (1337, 1337),
    'fmnistaug': (1337, 1337),
    'mnistaug': (1337, 1337),
    'fmnistefficient': (1337, 1337),
    'mnistefficient': (1337, 1337),
    'cifar10efficient': (1337, 1337),
    'svhnefficient': (1337, 1337)
}


def load_dataset(args):
    if args.dataset == 'gas':
        dataset = GAS('data/gas/ethylene_CO.pickle')
    elif args.dataset == 'bsds300':
        dataset = BSDS300('data/BSDS300/BSDS300.hdf5')
    elif args.dataset == 'hepmass':
        dataset = HEPMASS('data/hepmass')
    elif args.dataset == 'miniboone':
        dataset = MINIBOONE('data/miniboone/data.npy')
    elif args.dataset == 'power':
        dataset = POWER('data/power/data.npy')
    elif args.dataset == 'mnist':
        dataset = MNIST()
    elif args.dataset == 'fmnist':
        dataset = FASHIONMNIST()
    elif args.dataset == 'svhn':
        dataset = SVHN()
    elif args.dataset == 'cifar10':
        dataset = CIFAR10()
    elif args.dataset == 'svhngray':
        dataset = SVHNGRAYSCALE()
    elif args.dataset == 'cifar10gray':
        dataset = CIFAR10GRAYSCALE()
    elif args.dataset == 'fmnistaug':
        dataset = FASHIONMNISTAUG()
    elif args.dataset == 'mnistaug':
        dataset = MNISTAUG()
    elif args.dataset == 'fmnistefficient':
        dataset = FASHIONMNISTEFFICIENT()
    elif args.dataset == 'mnistefficient':
        dataset = MNISTEFFICIENT()
    elif args.dataset == 'cifar10efficient':
        dataset = CIFAR10GRAYSCALEEFFICIENT()
    elif args.dataset == 'svhnefficient':
        dataset = SVHNGRAYSCALEEFFICIENT()
    else:
        raise RuntimeError()

    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.trn.x).float().to(args.device))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_dim, shuffle=True)

    dataset_valid = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.val.x).float().to(args.device))
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_dim, shuffle=False)

    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.tst.x).float().to(args.device))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_dim, shuffle=False)
    
    args.n_dims = dataset.n_dims
    
    return data_loader_train, data_loader_valid, data_loader_test


def create_model(args, verbose=False):

    flows = []
    for f in range(args.flows):
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims))
            layers.append(Tanh())

        flows.append(
            BNAF(*([MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims), Tanh()] + \
                   layers + \
                   [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims)]),\
                 res=args.residual if f < args.flows - 1 else None
            )
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

    model = Sequential(*flows).to(args.device)
    params = sum((p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item()
                 for p in model.parameters()).item()
    
    if verbose:
        print('{}'.format(model))
        print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params, 
            NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))
                
    if args.save and not args.load:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params, 
                NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)
    
    return model
   
    
def load_model(model, optimizer, args, load_start_epoch=False):
    def f():
        print('Loading model..')
        checkpoint = torch.load(os.path.join(args.load or args.path, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if load_start_epoch:
            args.start_epoch = checkpoint['epoch']
            
    return f


def compute_log_p_x(model, x_mb):
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb

def negative_log_prob(model, x_mb):
    return -1*compute_log_p_x(model=model, x_mb=x_mb)

def load_args(args):
    file_args = open(os.path.join(args.load, 'args.json'), "r")
    contents = file_args.read()
    dictionary = json.loads(contents)
    #dictionary = eval(contents)
    for key, val in dictionary.items():
        if(key in ["save", "dataset"]):
            continue
        setattr(args, key, val)
    file_args.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='miniboone',
                        choices=['gas', 
                                 'bsds300',
                                 'hepmass',
                                 'miniboone',
                                 'power',
                                 'fmnist',
                                 'mnist',
                                 'svhn',
                                 'cifar10',
                                 'cifar10gray',
                                 'svhngray',
                                 'fmnistaug',
                                 'mnistaug',
                                 'fmnistefficient',
                                 'mnistefficient',
                                 'cifar10efficient',
                                 'svhnefficient'])

    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--batch_dim', type=int, default=200)
    parser.add_argument('--clip_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=1000)
    
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--cooldown', type=int, default=10)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=5e-4)
    parser.add_argument('--polyak', type=float, default=0.998)

    parser.add_argument('--flows', type=int, default=5)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=10)
    parser.add_argument('--residual', type=str, default='gated',
                       choices=[None, 'normal', 'gated'])

    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--tensorboard', type=str, default='tensorboard')
    parser.add_argument('--dataset_type', type=str, default="train")
    
    args = parser.parse_args()
    load_args(args)

    print('Arguments:')
    pprint.pprint(args.__dict__)

    print('Loading dataset..')
    args.batch_dim = 200
    data_loader_train, data_loader_valid, _ = load_dataset(args)
    if(args.dataset_type == "train"):
        data_loader = data_loader_train
    elif(args.dataset_type == "valid"):
        data_loader = data_loader_valid
    else:
        raise Exception()
    #if args.save and not args.load:
    #    print('Creating directory experiment..')
    #    print(args.path)
    #    os.mkdir(args.path)
    #    with open(os.path.join(args.path, 'args.json'), 'w') as f:
    #        json.dump(args.__dict__, f, indent=4, sort_keys=True)
    
    print('Creating BNAF model..')
    model = create_model(args, verbose=True)

    print('Creating optimizer..')
    optimizer = Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, polyak=args.polyak)
    
    print('Creating scheduler..')
    load_model(model, optimizer, args, load_start_epoch=True)()
    new_dir_path = f"./saved_plot_tensor_{str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')}/" 
    os.mkdir(path=new_dir_path)
    os.mkdir(path=f"{new_dir_path}{args.dataset}_{args.dataset_type}/")

    for index, (x,) in enumerate(data_loader):
        py_map = {"data":negative_log_prob(model, x)}
        torch.save(py_map, f"{new_dir_path}{args.dataset}_{args.dataset_type}/tensor_{index}")

    
    model = None


if __name__ == '__main__':
    main()
