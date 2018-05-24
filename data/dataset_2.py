"""
Divide the dataset into 3 parts, i.e. train set, val set and test set.
"""

from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class Driver(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, test=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test = test

        if self.test:
            with open(os.path.join(self.root, 'test.csv'), 'r') as f:
                lines = f.readlines()[1:]
            dataset = []
            for line in lines:
                dataset.append(line.strip().split(','))
        else:
            with open(os.path.join(self.root, 'train.csv'), 'r') as f:
                lines = f.readlines()[1:]
            dataset = []
            for line in lines:
                dataset.append(line.strip().split(','))

            num_train = int(0.7 * len(dataset))
            import random
            for _ in range(10):
                dataset = random.sample(dataset, len(dataset))
            if self.train:
                dataset = dataset[:num_train]
            else:
                dataset = dataset[num_train:]

        dataset = np.array(dataset)
        self.imgs = list(map(lambda x: os.path.join(self.root, x), dataset[:, 0]))
        self.target = list(map(int, dataset[:, 1]))

        if transform is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.test or (not self.train):
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(), normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.25, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.target[index]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    driver = Driver('/home/husencd/Downloads/dataset/driver', train=True)
    print(driver.__getitem__(1))  # random shuffle
    print(driver.__len__())  # 9083
    driver = Driver('/home/husencd/Downloads/dataset/driver', train=False, test=False)
    print(driver.__len__())  # 3894
    driver = Driver('/home/husencd/Downloads/dataset/driver', train=False, test=True)
    print(driver.__len__())  # 4331
