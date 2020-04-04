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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = arch_module.__dict__[args.arch](dataset=args.dataset)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location='cpu')
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
            datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, download=True, transform=transforms.Compose([
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
    pruning_plan = [
                    ('layer1.0',0.1), ('layer1.1',0.1), ('layer1.2',0.1), ('layer1.3',0.1), ('layer1.4',0.1), ('layer1.5',0.1), ('layer1.6',0.1), ('layer1.7',0.0), ('layer1.8',0.1),
                    ('layer2.0',0.0), ('layer2.1',0.2), ('layer2.2',0.2), ('layer2.3',0.2), ('layer2.4',0.2), ('layer2.5',0.2), ('layer2.6',0.2), ('layer2.7',0.2), ('layer2.8',0.2),
                    ('layer3.0',0.0), ('layer3.1',0.3), ('layer3.2',0.3), ('layer3.3',0.3), ('layer3.4',0.3), ('layer3.5',0.3), ('layer3.6',0.3), ('layer3.7',0.3), ('layer3.8',0.0)
                   ]
elif args.arch == 'resnet110':
    pruning_plan = [
                    ('layer1.0',0.1), ('layer1.1',0.1), ('layer1.2',0.1), ('layer1.3',0.1), ('layer1.4',0.1), ('layer1.5',0.1), ('layer1.6',0.1), ('layer1.7',0.1), ('layer1.8',0.1),
                    ('layer1.9',0.1), ('layer1.10',0.1), ('layer1.11',0.1), ('layer1.12',0.1), ('layer1.13',0.1), ('layer1.14',0.1), ('layer1.15',0.1), ('layer1.16',0.1), ('layer1.17',0.0),
                    ('layer2.0',0.0), ('layer2.1',0.2), ('layer2.2',0.2), ('layer2.3',0.2), ('layer2.4',0.2), ('layer2.5',0.2), ('layer2.6',0.2), ('layer2.7',0.2), ('layer2.8',0.2),
                    ('layer2.9',0.2), ('layer2.10',0.2), ('layer2.11',0.2), ('layer2.12',0.2), ('layer2.13',0.2), ('layer2.14',0.2), ('layer2.15',0.2), ('layer2.16',0.2), ('layer2.17',0.2), 
                    ('layer3.0',0.0), ('layer3.1',0.3), ('layer3.2',0.3), ('layer3.3',0.3), ('layer3.4',0.3), ('layer3.5',0.3), ('layer3.6',0.3), ('layer3.7',0.3), ('layer3.8',0.3), ('layer3.9',0.3),
                    ('layer3.10',0.3), ('layer3.11',0.3), ('layer3.12',0.3), ('layer3.13',0.3), ('layer3.14',0.3), ('layer3.15',0.3), ('layer3.16',0.3), ('layer3.17',0.3)
                   ]
elif args.arch == 'preresnet110':
    pruning_plan = [
                    ('layer1.0',0.1), ('layer1.1',0.1), ('layer1.2',0.1), ('layer1.3',0.1), ('layer1.4',0.1), ('layer1.5',0.1), ('layer1.6',0.1), ('layer1.7',0.1), ('layer1.8',0.1),
                    ('layer1.9',0.1), ('layer1.10',0.1), ('layer1.11',0.1), ('layer1.12',0.1), ('layer1.13',0.1), ('layer1.14',0.1), ('layer1.15',0.1), ('layer1.16',0.1), ('layer1.17',0.0),
                    ('layer2.0',0.0), ('layer2.1',0.2), ('layer2.2',0.2), ('layer2.3',0.2), ('layer2.4',0.2), ('layer2.5',0.2), ('layer2.6',0.2), ('layer2.7',0.2), ('layer2.8',0.2),
                    ('layer2.9',0.2), ('layer2.10',0.2), ('layer2.11',0.2), ('layer2.12',0.2), ('layer2.13',0.2), ('layer2.14',0.2), ('layer2.15',0.2), ('layer2.16',0.2), ('layer2.17',0.2), 
                    ('layer3.0',0.0), ('layer3.1',0.3), ('layer3.2',0.3), ('layer3.3',0.3), ('layer3.4',0.3), ('layer3.5',0.3), ('layer3.6',0.3), ('layer3.7',0.3), ('layer3.8',0.3), ('layer3.9',0.3),
                    ('layer3.10',0.3), ('layer3.11',0.3), ('layer3.12',0.3), ('layer3.13',0.3), ('layer3.14',0.3), ('layer3.15',0.3), ('layer3.16',0.3), ('layer3.17',0.3)
                   ]
elif args.arch == 'wrn_28_10':
    pruning_plan = [
                    ('block1.layer.0',0.1), ('block1.layer.1',0.1), ('block1.layer.2',0.1), ('block1.layer.3',0.1),
                    ('block2.layer.0',0.2), ('block2.layer.1',0.2), ('block2.layer.2',0.2), ('block2.layer.3',0.2),
                    ('block3.layer.0',0.3), ('block3.layer.1',0.3), ('block3.layer.2',0.3), ('block3.layer.3',0.3),
                   ]
elif args.arch == 'wrn_16_8':
    pruning_plan = [
                    ('block1.layer.0',0.1), ('block1.layer.1',0.1),
                    ('block2.layer.0',0.2), ('block2.layer.1',0.2),
                    ('block3.layer.0',0.3), ('block3.layer.1',0.3),
                   ]
else:
    raise ValueError("Expect arch to be one of [resnet56, resnet110, wrn_28_10] but got {}".format(args.arch))

def filter_pruning(model, pruning_plan):
    # get the cfg of pruned network
    cfg = []
    for block_name, prune_prob in pruning_plan:
        block = model.get_block(block_name)
        conv_layers = list(filter(lambda layer: isinstance(layer, nn.Conv2d), block.modules()))
        out_channels = conv_layers[0].out_channels
        num_keep = int(out_channels*(1-prune_prob))
        cfg.append(num_keep)
    # construct pruned network
    new_model = arch_module.__dict__[args.arch](dataset=args.dataset, cfg=cfg)
    # copy weight from original network to new network
    is_last_conv_pruned = False
    mask = None # mask of pruned layer
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            # current layer is not modified 
            if (m0.in_channels == m1.in_channels) and (m0.out_channels == m1.out_channels):
                m1.weight.data = m0.weight.data.clone()
                is_last_conv_pruned = False
            # remove the input weights corresponding to removed filter
            # Importance: as some layer could be pruned while having prior layer pruned as well
            # hence, it's crucial to set this condition above the m0.out_channels > m1.out_channels
            # as the is_last_conv_pruned flag would be set to False and can be rewrite if aforemention situation happend
            if m0.in_channels > m1.in_channels:
                # the filter would always
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[:, idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                is_last_conv_pruned = False 
            # current layer's filters are pruned
            # copy kept filter weight to new model
            if m0.out_channels > m1.out_channels:
                mask = create_l1_norm_mask(m0, m1.out_channels) 
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = m0.weight.data[idx.tolist(), :, :, :].clone()
                m1.weight.data = w.clone()
                is_last_conv_pruned = True 
        # adjust batchnorm with corresponding filter
        elif isinstance(m0, nn.BatchNorm2d):
            # if last conv layer is pruned then modify the batchnorm as well
            if is_last_conv_pruned:
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                m1.weight.data = m0.weight.data[idx.tolist()].clone()
                m1.bias.data = m0.bias.data[idx.tolist()].clone()
                m1.running_mean = m0.running_mean[idx.tolist()].clone()
                m1.running_var = m0.running_var[idx.tolist()].clone()
            # if the last conv layer wasn't modified then simply copy weights
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        # linear layer will not be pruned
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
    return new_model

def create_l1_norm_mask(layer, num_keep):
    """
        Create a 1d tensor with binary value, the i-th element is set to 1
            if i-th output filter of this layer is kept, 0 otherwise
            selection criterion - l1 norm based
        :param layer: nn.Conv2d
        :param num_keep: int - number of filters that would be keep
        :return: 1D torch.tensor - mask
    """
    out_channels = layer.out_channels
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    L1_norm = np.sum(weight_copy, axis=(1,2,3))
    arg_max = np.argsort(L1_norm)
    arg_max_rev = arg_max[::-1][:num_keep]

    # create mask
    mask = torch.zeros(out_channels)
    mask[arg_max_rev.tolist()] = 1
    
    return mask 

if __name__ == '__main__':
    newmodel = filter_pruning(model, pruning_plan)
    torch.save({
                'cfg': newmodel.cfg,
                'state_dict': newmodel.state_dict()
               },
               os.path.join(args.save, 'pruned.pth.tar'))
    print(newmodel)
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    model = newmodel
    acc = test(model)
    print('cfg: ', str(model.cfg))
    print("number of parameters: "+str(num_parameters))
    with open(os.path.join(args.save, "prune.txt"), "w") as fp:
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc)+"\n")
