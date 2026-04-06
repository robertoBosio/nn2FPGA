import torch
import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from coco import CocoDetection
from coco import preproc as coco_preproc

class ImageNet(Dataset):
    def __init__(self, root, train, transform=None, sample_size=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        items = os.listdir(root + "/train/")
        sorted_items = sorted(items)
        for class_id, syn_id in enumerate(sorted_items):
            self.syn_to_class[syn_id] = class_id

        if train:
            image_path = root + "/train/"
        else:
            image_path = root + "/val/"
        items = os.listdir(image_path)
        sorted_items = sorted(items)
        for syn_id in sorted_items:
            syn_folder = os.path.join(image_path, syn_id)
            class_id = self.syn_to_class[syn_id]
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(class_id)
        
        if sample_size is not None:
            # Randomly sample a subset of the dataset and targets
            assert len(self.samples) == len(self.targets)
            indices = np.random.choice(len(self.samples), sample_size, replace=False)
            self.samples = [self.samples[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]

    def __len__(self):
            return len(self.samples)
    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            x = self.transform(x)
            return x, self.targets[idx]

def cifar10_dataloader(batch_size):
    CIFAR10_DIRECTORY = '/home/datasets/cifar10'

    if not os.path.exists(CIFAR10_DIRECTORY):
        print("CIFAR10 Dataset not present")
        exit(0)

    test_data = torchvision.datasets.CIFAR10(
        CIFAR10_DIRECTORY,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.float(),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = 32
    IW = 32
    OCH = 10
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim

def cifar10_4bit_dataloader(batch_size):
    CIFAR10_DIRECTORY = '/home/datasets/cifar10'

    if not os.path.exists(CIFAR10_DIRECTORY):
        print("CIFAR10 Dataset not present")
        exit(0)

    test_data = torchvision.datasets.CIFAR10(
        CIFAR10_DIRECTORY,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.float(),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = 32
    IW = 32
    OCH = 10
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim

def coco_dataloader(batch_size):
    COCO_DIRECTORY = '/home/datasets/coco/'

    if not os.path.exists(COCO_DIRECTORY):
        print("COCO Dataset not present")
        exit(0)

    images_path = os.path.join(COCO_DIRECTORY, 'images')
    annotations_path = os.path.join(COCO_DIRECTORY, 'annotations')

    X_AXIS = 360
    Y_AXIS = 640

    test_data = CocoDetection(
        images_path,
        os.path.join(annotations_path, 'instances_val2017.json'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.float(),
            lambda x: coco_preproc(x, Y_AXIS, X_AXIS),
        ])
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    ICH = 3
    IH = Y_AXIS
    IW = X_AXIS
    OCH = 16
    MAX_NUM_BOX = 3000

    buffer_dim = [
        ICH*IW*IH,
        OCH*MAX_NUM_BOX
    ]

    return test_loader, buffer_dim

def vw_dataloader(batch_size):
    IMAGE_SIZE = 96
    BASE_DIR = os.path.join("/home/datasets/vw", 'vw_coco2014_96')
    vw_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_args = {
        'transform': vw_transform,
        'root': BASE_DIR
    }
    dataset = torchvision.datasets.ImageFolder

    dataset = dataset(**train_args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    buffer_dim = [
        3*IMAGE_SIZE*IMAGE_SIZE,
        2
    ]

    return loader, buffer_dim

def imagenet_dataloader(batch_size):
    IMAGENET_DIRECTORY = '/home/datasets/Imagenet/'

    if not os.path.exists(IMAGENET_DIRECTORY):
        print("IMAGENET Dataset not present")
        exit(0)

    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args = {
        'train': False,
        'transform': transform,
        'root': IMAGENET_DIRECTORY,
        'sample_size': None
    }

    dataset = ImageNet(**args)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    ICH = 3
    IH = 224
    IW = 224
    OCH = 1000
    OH = 1
    OW = 1

    buffer_dim = [
        ICH*IW*IH,
        OCH*OH*OW
    ]

    return test_loader, buffer_dim