import torch
import torch.nn as nn

import os
from models import resnet

model_path = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def get_model_param(args):
    # assert args.model in ['resnet', 'vgg']

    if args.model == 'resnet':
        assert args.model_depth in [18, 34, 50, 101, 152]

        from models.resnet import get_fine_tuning_parameters

        if args.model_depth == 18:
            model = resnet.resnet18(pretrained=False, input_size=args.input_size, num_classes=args.n_classes)
        elif args.model_depth == 34:
            model = resnet.resnet34(pretrained=False, input_size=args.input_size, num_classes=args.n_classes)
        elif args.model_depth == 50:
            model = resnet.resnet50(pretrained=False, input_size=args.input_size, num_classes=args.n_classes)
        elif args.model_depth == 101:
            model = resnet.resnet101(pretrained=False, input_size=args.input_size, num_classes=args.n_classes)
        elif args.model_depth == 152:
            model = resnet.resnet152(pretrained=False, input_size=args.input_size, num_classes=args.n_classes)

    # elif args.model == 'vgg':
    #     pass

    # Load pretrained model here
    if args.finetune:
        pretrained_model = model_path[args.arch]
        args.pretrain_path = os.path.join(args.root_path, 'pretrained_models', pretrained_model)
        print("=> loading pretrained model '{}'...".format(pretrained_model))

        model.load_state_dict(torch.load(args.pretrain_path))

        # Only modify the last layer
        if args.model == 'resnet':
            model.fc = nn.Linear(model.fc.in_features, args.n_finetune_classes)
        # elif args.model == 'vgg':
        #     pass

        parameters = get_fine_tuning_parameters(model, args.ft_begin_index, args.lr_mult1, args.lr_mult2)
        return model, parameters

    return model, model.parameters()
