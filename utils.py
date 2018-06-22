import csv
import torch


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


class Logger(object):
    """Outputs log files"""
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # batch_size x maxk
        pred = pred.t()  # transpose, maxk x batch_size
        # target.view(1, -1): convert (batch_size,) to 1 x batch_size
        # expand_as: convert 1 x batch_size to maxk x batch_size
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # maxk x batch_size

        res = []
        for k in topk:
            # correct[:k] converts "maxk x batch_size" to "k x batch_size"
            # view(-1) converts "k x batch_size" to "(k x batch_size,)"
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
