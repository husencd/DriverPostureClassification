import time

from utils import AverageMeter, calculate_accuracy


def test(data_loader, model, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
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

        # measure accuracy and record loss
        prec1, prec3 = calculate_accuracy(output, target, topk=(1, 3))
        # Attention: prec1[0], convert torch.Size([1]) to torch.Size([])
        top1.update(prec1[0].item(), input.size(0))
        top3.update(prec3[0].item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % args.log_interval == 0:
            print('Test Iter [{0}/{1}]\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      top1=top1,
                      top3=top3,
                      batch_time=batch_time,
                      data_time=data_time))

    print(' * Prec@1 {top1.avg:.2f}% | Prec@3 {top3.avg:.2f}%'.format(
        top1=top1, top3=top3))

    return top1.avg
