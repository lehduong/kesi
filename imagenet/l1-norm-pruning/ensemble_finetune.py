import argparse
import os
import numpy as np
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import sys
import models as arch_module

sys.path.append('../')
from data_loader import TinyImageNet
from functools import reduce
from utils.losses import KLDivergenceLoss

model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 25)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--temperature', type=float, default=5, metavar='N',
                    help='temperature for knowledge distillation (default: 5)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--use-onecycle', dest='use_onecycle', action='store_true')
parser.add_argument('--no-onecycle', dest='use_onecycle', action='store_false')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--s', type=float, default=0,
                    help='scale sparse rate (default: 0)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='the PATH to pruned model')
parser.add_argument('--num_classes', default=200, type=int, help='number of classes of classifier')

parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

parser.add_argument('--schedule', type=int, nargs='+', default=[13, 19],
                    help='Decrease learning rate at these epochs.')
parser.set_defaults(use_onecycle=True)

best_prec1 = 0

checkpoint_paths = [
    "checkpoints/model_best.pth.tar",
    "prune_1/checkpoint.pth.tar",
    "prune_2/checkpoint.pth.tar",
    "prune_3/checkpoint.pth.tar",
    "prune_4/checkpoint.pth.tar",
]

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # load models:
    if args.refine:
        # load student
        print("=> loading checkpoint '{}'".format(args.refine))
        checkpoint = torch.load(args.refine, map_location='cpu')
        cfg = checkpoint['cfg'] if 'cfg' in checkpoint.keys() else None
        model = arch_module.__dict__[args.arch](num_classes=args.num_classes, cfg=cfg)
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        for param in model.parameters():
            param.requires_grad = True
        # load teachers
        models = list()
        for checkpoint_path in checkpoint_paths:
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            cfg = checkpoint['cfg'] if 'cfg' in checkpoint.keys() else None
            tmp = arch_module.__dict__[args.arch](num_classes=args.num_classes, cfg=cfg)
            tmp = nn.DataParallel(tmp).cuda()
            tmp.load_state_dict(checkpoint['state_dict'])
            tmp.eval()
            for param in tmp.parameters():
                param.requires_grad = False 
            models.append(tmp)
            best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint.keys() else checkpoint['acc']
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(checkpoint_path, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError('Please specify refined model')

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225))

    training_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # For Tiny ImageNet dataset
    train_dataset = TinyImageNet(args.data, 'train', transform=training_transform, in_memory=False)
    val_dataset = TinyImageNet(args.data, 'val', transform=valid_transform, in_memory=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and optimizer
    criterion = KLDivergenceLoss(temperature=args.temperature)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_onecycle:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, div_factor=10,
                                                           epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.1,
                                                           final_div_factor=100)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler, gamma=args.gamma)
    
    if args.evaluate:
        validate(val_loader, model, models, criterion)
        return

    history_score = np.zeros((args.epochs + 1, 1))
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    for epoch in range(args.start_epoch, args.epochs):
        lr = next(iter(optimizer.param_groups))['lr']
        print("EPOCH: {} lr: {} ".format(epoch, lr))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, models, criterion, optimizer, epoch, lr_scheduler)
        # evaluate on validation set
        prec1 = validate(val_loader, model, models, criterion)
        history_score[epoch] = prec1.cpu()
        np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1).cpu()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'cfg': model.module.cfg
        }, is_best, args.save)

    history_score[-1] = best_prec1
    np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')


def train(train_loader, model, models, criterion, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.cuda(), target.cuda()
        # compute output
        output = model(data)
        output_tc = list()
        with torch.no_grad():
            for model_tc in models:
                output_tc.append(model_tc(data))
        loss = reduce(lambda acc, elem: acc + criterion(output, elem), output_tc, 0)/len(models)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
        lr_scheduler.step()

def validate(val_loader, model, models, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_ens = AverageMeter()
    top5_ens = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()

            # compute output
            output = model(data)
            output_tc = list()
            with torch.no_grad():
                for model_tc in models:
                    output_tc.append(model_tc(data))
            loss = reduce(lambda acc, elem: acc + criterion(output, elem), output_tc, 0)/len(models)
            
            # compute output of ensemble
            output_ens = torch.zeros_like(output)
            for model_tc in models:
                tmp = model_tc(data)
                output_ens += softmax(tmp)
            output_ens /= len(models)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure accuracy of ensemble
            prec1, prec5 = accuracy(output_ens.data, target, topk=(1, 5))
            top1_ens.update(prec1[0], data.size(0))
            top5_ens.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'Prec@1 ensemble {top1_ens.val:.3f} ({top1_ens.avg:.3f})'
                    'Prec@5 ensemble {top5_ens.val:.3f} ({top5_ens.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5, top1_ens=top1_ens, top5_ens=top5_ens))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Ensemble prec@1 {top1_ens.avg:.3f} Ensemble prec@5 {top5_ens.avg:.3f}'
            .format(top1=top1, top5=top5, top1_ens=top1_ens, top5_ens=top5_ens))

    return top1.avg


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()