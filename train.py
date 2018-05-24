import torch
import time
import os

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, args, device,
                epoch_logger, batch_logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end_time = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end_time)

        input = input.to(device)
        target = target.to(device)

        # compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = calculate_accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        # prec1[0]: convert torch.Size([1]) to torch.Size([])
        top1.update(prec1[0].item(), input.size(0))
        top3.update(prec3[0].item(), input.size(0))
        """
        a = np.array([1, 2, 3])
        b = torch.from_numpy(a)  # tensor([ 1,  2,  3])
        c = b.sum()  # tensor(6)
        d = b.sum(0)  # tensor(6)
        e = b.sum(0, keepdim=True)  # tensor([ 6]), torch.Size([1])
        e[0]  # tensor(6), torch.Size([])
        e.item()  # 6
        """

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % args.log_interval == 0:
            print('Train Epoch [{0}/{1}]([{2}/{3}])\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'LR {lr:f}\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                      epoch,
                      args.epochs,
                      i + 1,
                      len(data_loader),
                      loss=losses,
                      top1=top1,
                      top3=top3,
                      lr=optimizer.param_groups[0]['lr'],
                      batch_time=batch_time,
                      data_time=data_time))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'top1': top1.val,
            'top3': top3.val,
            'lr': optimizer.param_groups[0]['lr']
        })

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'top1': top1.avg,
        'top3': top3.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % args.checkpoint_interval == 0:
        save_file_path = os.path.join(args.checkpoint_path, 'save_{}_{}.pth'.format(args.arch, epoch))
        states = {
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    vis.plot('Train loss', losses.avg)
    vis.plot('Train accu', top1.avg)
    vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, accu:{accu}".format(
        epoch=epoch,
        lr=optimizer.param_groups[0]['lr'],
        loss=losses.avg,
        accu=top1.avg))


def val_epoch(epoch, data_loader, model, criterion, args, device, epoch_logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end_time = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end_time)

        input = input.to(device)
        target = target.to(device)

        # compute output and loss
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = calculate_accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))
        top3.update(prec3[0].item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % args.log_interval == 0:
            print('Valid Epoch [{0}/{1}]([{2}/{3}])\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                      epoch,
                      args.epochs,
                      i + 1,
                      len(data_loader),
                      loss=losses,
                      top1=top1,
                      top3=top3,
                      batch_time=batch_time,
                      data_time=data_time))

    print(' * Prec@1 {top1.avg:.2f}% | Prec@3 {top3.avg:.2f}%'.format(
        top1=top1, top3=top3))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'top1': top1.avg,
        'top3': top3.avg
    })

    vis.plot('Val loss', losses.avg)
    vis.plot('Val accu', top1.avg)

    return losses.avg, top1.avg
