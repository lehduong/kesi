from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import models as model_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from functools import reduce
from utils.losses import KLDivergenceLoss


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')

checkpoint_paths = ['checkpoints/model_best.pth.tar', 
                    'prune_1/checkpoint.pth.tar',
                    'prune_2/checkpoint.pth.tar',
                    'prune_3/checkpoint.pth.tar',
                    'prune_4/checkpoint.pth.tar',
                    'prune_5/checkpoint.pth.tar',
                    ]

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = model_module.__dict__[args.arch](dataset=args.dataset)
models = []

if args.refine:
    print("=> loading checkpoint '{}'".format(args.refine))
    checkpoint = torch.load(args.refine)
    model = model_module.__dict__[args.arch](dataset=args.dataset, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
      param.requires_grad = True 
    for path in checkpoint_paths:
        print("=> loading ensemble '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        tmp = model_module.__dict__[args.arch](dataset=args.dataset, cfg=checkpoint['cfg'])
        tmp.load_state_dict(checkpoint['state_dict'])
        tmp.eval()
        for param in tmp.parameters():
            param.requires_grad = False  
        models.append(tmp)
        best_prec1 = checkpoint['acc'] if 'acc' in checkpoint.keys() else checkpoint['best_prec1']
        print("=> loaded model '{}' (epoch {}) Prec1: {:f}"
              .format(path, checkpoint['epoch'], best_prec1))

if args.cuda:
    model.cuda()
    for m in models:
        m.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=[int(0.5*args.epochs), int(0.75*args.epochs)],
                                              gamma=0.2)

criterion = KLDivergenceLoss(temperature=5)

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    lr = next(iter(optimizer.param_groups))['lr']
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        optimizer.zero_grad()
        output_tc = []
        with torch.no_grad():
            for model_tc in models:
                output_tc.append(model_tc(data))
        loss = reduce(lambda acc, elem: acc + criterion(output, elem), output_tc, 0)/len(models) 
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() 
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    lr_scheduler.step()

def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_loss_ens = 0
    correct_ens = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)
            
            # ensemble predictions
            output_ens = torch.zeros_like(output)
            for m in models:
                tmp_output = m(data) 
                output_ens += tmp_output
            test_loss_ens += F.cross_entropy(output_ens, target, size_average=False) # sum up batch loss
            pred_ens = output_ens.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_ens += pred_ens.eq(target.view_as(pred)).cpu().sum()
            test_loss_ens /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Ensemble Accuracy: {:.2f}%\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            100. * correct_ens / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)
