import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST
from utils.htru1 import HTRU1
from utils.mydataset_HTRU import *

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot,  split='test', download=args.download, transform=trans)

    elif args.dataset == 'htru':
        trans = transforms.Compose([
            # transforms.Resize((64, 64)),
            # transforms.Lambda(lambda img: img.crop((0, 10, 64, 64))),  # 裁剪掉上方 10 像素
            transforms.Resize((64, 64)),  # 再次调整大小到 64x64
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = HTRU1('data', train=True, download=True, transform=trans)
        train_dataset = select_sample(train_data , true_num=1000 , false_num=0)
        test_data = HTRU1('data', train=False, download=True, transform=trans)
        test_dataset = select_sample(test_data , true_num=500 , false_num=0)
    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
