from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get(dataset_type="mnist"):
    # MINST
    if type.lower() == "mnist":

        train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        input_size = 28 * 28

    return train_loader, test_loader, input_size
