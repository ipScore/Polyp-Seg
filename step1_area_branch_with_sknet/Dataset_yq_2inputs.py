from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F


class Dataset_unet(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_ids = os.listdir(image_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        label_name = self.image_ids[idx]

        image = Image.open(self.image_dir + image_name).convert('RGB')
        label = Image.open(self.label_dir + label_name).convert('L')

        # return dictionary
        sample = {'image': image, 'label': label, 'num': image_name}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


if __name__ == '__main__':
    image_dir = '../train/images/'
    label_dir = '../train/labels/'

    transform1 = transforms.Compose([ToTensor()])
    dataset = Dataset_unet(image_dir, label_dir, transform=transform1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
        print('i_batch:{}'.format(i_batch))

