import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import json
from args import parse_args
from model import get_model_param
from data import Driver
from utils import Logger
from tools import Visualizer
from train import train_epoch, val_epoch
import test

best_prec1 = 0
best_epoch = 1


def main():
    global args, best_prec1, best_epoch
    args = parse_args()

    if args.root_path != '':
        args.result_path = os.path.join(args.root_path, args.result_path)
        args.checkpoint_path = os.path.join(args.root_path, args.checkpoint_path)
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
        if not os.path.exists(args.checkpoint_path):
            os.mkdir(args.checkpoint_path)
        if args.resume_path:
            args.resume_path = os.path.join(args.checkpoint_path, args.resume_path)

    args.arch = '{}{}'.format(args.model, args.model_depth)

    torch.manual_seed(args.manual_seed)

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # create model
    model, parameters = get_model_param(args)
    print(model)
    model = model.to(device)

    with open(os.path.join(args.result_path, 'args.json'), 'w') as args_file:
        json.dump(vars(args), args_file)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.001, patience=args.lr_patience)

    lr_mult = []
    for param_group in optimizer.param_groups:
        lr_mult.append(param_group['lr'])

    # optionally resume from a checkpoint
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'...".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.begin_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_path))

    if args.train:
        train_dataset = Driver(root=args.data_path, train=True, test=False)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)
        train_logger = Logger(
            os.path.join(args.result_path, 'train.log'),
            ['epoch', 'loss', 'top1', 'top3', 'lr'])
        train_batch_logger = Logger(
            os.path.join(args.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'top1', 'top3', 'lr'])
    if args.val:
        val_dataset = Driver(root=args.data_path, train=False, test=True)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers)
        val_logger = Logger(
            os.path.join(args.result_path, 'val.log'),
            ['epoch', 'loss', 'top1', 'top3'])

    print('=> Start running...')
    vis = Visualizer(env=args.env)
    for epoch in range(args.begin_epoch, args.epochs + 1):
        if args.train:
            adjust_learning_rate(optimizer, epoch, lr_mult)
            train_epoch(epoch, train_loader, model, criterion, optimizer, args, device, train_logger, train_batch_logger, vis)
            print()

        if args.val:
            val_loss, val_prec1 = val_epoch(epoch, val_loader, model, criterion, args, device, val_logger, vis)
            print()
            # remember best prec@1 and save checkpoint
            if val_prec1 > best_prec1:
                best_prec1 = val_prec1
                best_epoch = epoch
                print('=> Saving current best model...\n')
                save_file_path = os.path.join(args.result_path, 'save_best_{}_{}.pth'.format(args.arch, epoch))
                state = {
                    'epoch': best_epoch,
                    'arch': args.arch,
                    'best_prec1': best_prec1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, save_file_path)

        # if args.train and args.val:
            # scheduler.step(val_loss)

    if args.test:
        test_dataset = Driver(root=args.data_path, train=False, test=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers)
        # # if you only test the model, you need to set the "best_epoch" manually
        # best_epoch = 10  # set manually
        saved_model_path = os.path.join(args.result_path, 'save_best_{}_{}.pth'.format(args.arch, best_epoch))
        print("Using '{}' for test...".format(saved_model_path))
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        test.test(test_loader, model, args, device)


def adjust_learning_rate(optimizer, epoch, lr_mult):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**((epoch - 1) // 30))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_mult[i]


if __name__ == '__main__':
    main()
