import os.path as osp
import sys

sys.path.append(osp.dirname(sys.path[0]))

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from neko import neko_utils


def get(dataset_type, dataset_path, batch_size):
    # MINST
    if dataset_type.lower() == "mnist":
        neko_utils.mkdir_nf(dataset_path)
        train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        input_size = 28 * 28

        return train_loader, test_loader, input_size

    return None
