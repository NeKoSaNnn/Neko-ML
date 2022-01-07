import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import utils

utils = utils()


def get(dataset_type, dataset_path, batch_size, resize=None, num_workers=utils.get_num_workers()):
    trans = [transforms.ToTensor()]
    if resize:
        trans.append(transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # MINST
    if dataset_type.lower() == "mnist":
        utils.mkdir_nf(dataset_path)
        train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=trans, download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=trans, download=True)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

        input_size = 28 * 28

        return train_loader, test_loader, input_size

    return None
