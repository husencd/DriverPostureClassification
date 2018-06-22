#!/usr/bin/env sh

echo "Download resnet models pretrained on ImageNet..."

wget -N https://download.pytorch.org/models/resnet18-5c106cde.pth
wget -N https://download.pytorch.org/models/resnet34-333f7ec4.pth
wget -N https://download.pytorch.org/models/resnet50-19c8e357.pth
wget -N https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
wget -N https://download.pytorch.org/models/resnet152-b121ed2d.pth