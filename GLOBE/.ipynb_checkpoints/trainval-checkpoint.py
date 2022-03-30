import torch
import numpy as np
from tqdm import tqdm

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch GLOBE Training')
    # environ settings
    parser.add_argument('--dname', default='Pancreas', type=str,
                        help='dataset name')
    parser.add_argument('--mode', default='unsupervised', type=str,
                        help='supervised or unsupervised')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--workers', default=10, type=int,
                        help='number of worker for dataloader')

    # prep params
    parser.add_argument('--select_hvg', default=True, type=int,
                        help='whether to select hvgs'
                        )
    parser.add_argument('--scale', default=False, type=int,
                        help='whether scale per batch')

    # model arch settings
    parser.add_argument('--block_level', default=1, type=int,
                        help='which block to choose, 0-naive block, 1-dropout block, 2-batchnorm block')
    parser.add_argument('--lat_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--proj_dim', default=64, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--temp', default=0.7, type=float,
                        help='temperature')
    parser.add_argument('--header', default='mlp', type=str,
                        help='use mlp head')
    parser.add_argument('--batchnorm', default=0, type=int, 
                        help='batchnorm level, 0: no, 1: encoder, 2: encoder+projection')
    parser.add_argument('--dropout_rate', default=0.3, type=float,
                        help='dropout rate for hidden layers')
    parser.add_argument('--init', default='uniform', type=str,
                        help='weights initial method')

    # training params
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--adjustLr', action='store_true', 
                        help='whether adjust learning rate during learning')
    parser.add_argument('--cos', action='store_true', 
                        help='cosine learning rate schedule')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum of SGD solver')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-4)')


    parser.add_argument('--epochs', default=80, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')

    # logging settings
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='print frequency (default: 10)')


    args = parser.parse_args()
    return args


def supervised_train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    pass
    # after publishment

# only used for MNN+InfoNCE, damn it
def supervised_train_one_epoch_withNCE(train_loader, model, criterion, optimizer, epoch):
    # after publishment
    pass 


def unsupervised_train_one_epoch(pos_graph, view='double'):
    # after publishment
    pass  


def unsupervised_train_one_epoch_withNCE(train_loader, model, criterion, optimizer, epoch):
    # after publishment
    pass 
