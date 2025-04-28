import torchvision
from torchvision.tv_tensors import Image
import torch

class CIFAR10(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = "/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/cifar10",
                 transform = None, **kwargs):

        self.cifar10 = torchvision.datasets.CIFAR10(root=root)
        self.transform = transform

    def __getitem__(self, index: int):
        image, target = self.cifar10[index]
        item = {
            'image': Image(image),
            'target': target,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self.cifar10)