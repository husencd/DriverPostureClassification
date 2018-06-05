import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Driver Posture Classification')

    # path
    parser.add_argument('--data_path', default='/home/husencd/Downloads/dataset/driver', type=str,
                        help='Driver data directory path')
    parser.add_argument('--root_path', default='/home/husencd/husen/pytorch/learn/DriverPostureClassification', type=str,
                        help='Project root directory path')
    parser.add_argument('--result_path', default='results', type=str,
                        help='Result directory path')
    parser.add_argument('--checkpoint_path', default='checkpoints', type=str,
                        help='Checkpoint directory path (snapshot)')
    parser.add_argument('--resume_path', default='', type=str,
                        help='Saved model (checkpoint) path of previous training')

    # I/O
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input size of image')
    parser.add_argument('--n_classes', default=1000, type=int,
                        help='Number of classes (ImageNet: 1000,)')
    parser.add_argument('--n_finetune_classes', default=10, type=int,
                        help='Number of classes for fine-tuning, n_classes is set to the number when pre-training')

    # about model configuration
    parser.add_argument('--model', default='resnet', type=str,
                        help='(vgg | resnet | resnext | densenet)')
    parser.add_argument('--model_depth', default=18, type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101 | 152)')

    # about optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_mult1', default=0.1, type=float,
                        help='Multiplication factor of learning rate in those pre-trained layers')
    parser.add_argument('--lr_mult2', default=1, type=float,
                        help='Multiplication factor of learning rate in those newly-created layers')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')

    # train, val, test, fine-tune
    parser.add_argument('--train', action='store_true', default=True,
                        help='If true, training is performed.')
    parser.add_argument('--val', action='store_true', default=True,
                        help='If true, validation is performed.')
    parser.add_argument('--test', action='store_true', default=True,
                        help='If true, test is performed.')
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='If True, fine-tune on a model that has been pretrained on ImageNet')
    parser.add_argument('--ft_begin_index', default=0, type=int,
                        help='Begin block index of fine-tuning')

    # batch size and epoch
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch Size')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='Test batch Size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')

    # training log and checkpoint
    parser.add_argument('--log_interval', default=10, type=int,
                        help='How many batches to wait before logging training status')
    parser.add_argument('--checkpoint_interval', default=20, type=int,
                        help='Trained model is saved at every this epochs.')

    # about device
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='If False, cuda is not used.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of threads for multi-thread loading')

    # random number seed
    parser.add_argument('--manual_seed', default=1, type=int,
                        help='Manually set random seed')

    # visdom
    parser.add_argument('--env', default='default', type=str,
                        help='Visdom enviroment')

    args = parser.parse_args()

    return args
