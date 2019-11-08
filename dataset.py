import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IDataset(Dataset):

    def __init__(self, imgs_path, transform=None):
        self.root = imgs_path
        self.fnames = os.listdir(imgs_path)
        self.len = len(self.fnames)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path = os.path.join(self.root, self.fnames[item])
        image = Image.open(path)

        if self.transform is None:
            image = transforms.ToTensor()(image)
        else:
            image = self.transform(image)

        return image
