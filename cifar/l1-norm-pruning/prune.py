import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models as arch_module

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--arch', type=str, default='resnet56',
                    help='depth of the resnet')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = arch_module.__dict__[args.arch](dataset=args.dataset)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model = arch_module.__dict__[args.arch](dataset=args.dataset, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("No checkpoint found at '{}'".format(args.model))

if args.cuda:
    model.cuda()

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

if args.arch == 'resnet56':
    skip = {
        'A': [16, 20, 38, 54],
        'B': [16, 18, 20, 34, 38, 54],
    }
    prune_prob = {
        'A': [0.1, 0.2, 0.3],
        'B': [0.6, 0.3, 0.1],
    }
    stages = [18, 36]
elif args.arch == 'resnet110':
    skip = {
        'A': [36, 38, 74],
        'B': [36, 38, 74],
    }
    prune_prob = {
        'A': [0.1, 0.2, 0.3],
        'B': [0.5, 0.4, 0.3],
    }
    stages = [36, 72] 
elif args.arch == 'wrn':
    skip = {
        'A': [4,],
        'B': []
    },
    prune_prob = {
        'A': [0.1, 0.2, 0.3],
        'B': [0.5, 0.4, 0.3],
    }
    stages = []
else:
    raise ValueError("Expect arch to be either resnet56 or resnet110 but got {}".format(args.arch))

def construct_result_model(model, skip, prune_prob, stages):
    """
        Create the config for result network
        :param model: nn.Module - model that would be pruned
        :param skip: list - list of index of layer that would be skiped
        :param prune_prob: list - the probability of weights that would be pruned of each stages. \
            Note that resnet models for cifar have 3 stages, hence, the length of this list in \
            that case would be 3
        :param stages: list - indexes of first layer of each stages
        :return: cfg - list - the config that can be passed to constructor of models\
                 cfg_mask - List of Tensor, each element is a tensor which shape equals to num_ouput of
                            identical layers. 1 indicate corresponding filter is kept 0 otherwise.
    """
    layer_id = 1
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            # if current layer is in skip list the simply add the num_output to config 
            if layer_id in skip[args.v]:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                layer_id += 1
                continue
            # layer not in the skip list + layer's index is even => pruned this layer filter 
            if layer_id % 2 == 0:
                # first determine the stage of current layer to pick the right pruning ratio
                if layer_id <= stages[0]:
                    stage = 0
                elif layer_id <= stages[1]:
                    stage = 1
                else:
                    stage = 2
                prune_prob_stage = prune_prob[args.v][stage]
                
                # filter selection
                #TODO: update other filter selecting criterions 
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1,2,3))
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]
                
                # create mask 
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                
                # add mask and num_filter to cfg
                cfg_mask.append(mask)
                cfg.append(num_keep)
                layer_id += 1
                continue
            # ignore 
            layer_id += 1
    return cfg, cfg_mask

def filter_prune(model, skip, prune_prob, stages):
    """
        Run filter pruning for given network
        :param model: nn.Module - model that would be pruned
        :param skip: list - list of index of layer that would be skiped
        :param prune_prob: list - the probability of weights that would be pruned of each stages. \
            Note that resnet models for cifar have 3 stages, hence, the length of this list in \
            that case would be 3
        :param stages: list - indexes of first layer of each stages
        :return: nn.Module - pruned network
    """
    # get new config
    cfg, cfg_mask = construct_result_model(model, skip, prune_prob, stages)
    # create compact model with above config
    newmodel = arch_module.__dict__[args.arch](dataset=args.dataset, cfg=cfg)
    if args.cuda:
        newmodel.cuda()

    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    conv_count = 1
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            # layer hasn't been pruned 
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                conv_count += 1
                continue
        elif isinstance(m0, nn.BatchNorm2d):
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg-1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
                continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
    
    return newmodel

if __name__ == '__main__':
    newmodel = filter_prune(model, skip, prune_prob, stages)
    torch.save({
                'cfg': newmodel.cfg, 
                'state_dict': newmodel.state_dict()
               }, 
               os.path.join(args.save, 'pruned.pth.tar'))
    print(newmodel)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    model = newmodel
    acc = test(model)

    print("number of parameters: "+str(num_parameters))
    with open(os.path.join(args.save, "prune.txt"), "w") as fp:
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc)+"\n")