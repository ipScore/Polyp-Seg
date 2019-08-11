import torchvision.transforms.functional as F
from torchvision import transforms
import scipy.ndimage
import random
from PIL import Image
import numpy as np
import cv2
from skimage import transform as tf
from Dataset_yq_2inputs import Dataset_unet
from torch.utils.data import DataLoader


class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label), 'num': sample['num']}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}
        
        return {'image': image, 'label': label}


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, np.numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, sample):

        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        image, label = sample['image'], sample['label']

        if random.random() < 0.5:
            angle = self.get_params(self.degrees)
            return {'image': F.rotate(image, angle, self.resample, self.expand, self.center), 'label': F.rotate(label, angle, self.resample, self.expand, self.center)}

        return {'image': image, 'label': label}


class RandomRotation_yq(object): 
    def __init__(self, degrees):
        self.degrees = degrees  # e.g. 80

    def __call__(self, sample):        
        image, label = sample['image'], sample['label']

        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            
            random_degree = random.uniform(-self.degrees, self.degrees)
            rotate_image = scipy.ndimage.rotate(image, random_degree, reshape=False)
            rotate_label = scipy.ndimage.rotate(label, random_degree, reshape=False)
            
            rotate_image = Image.fromarray(rotate_image.astype('uint8'), 'RGB')
            rotate_label = Image.fromarray(rotate_label.astype('uint8'), 'L')
            return {'image': rotate_image, 'label': rotate_label}
        
        return {'image': image, 'label': label}


class Rescale(object):

    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < 0.5:
            origin_length = image.size[0]
            new_length = int(origin_length * self.scale)
            return {'image': image.resize((new_length, new_length), Image.ANTIALIAS), 'label': label.resize((new_length, new_length), Image.ANTIALIAS)}
        
        return {'image': image, 'label': label}


class RandomZoom(object):
    def __init__(self, zoom):
        self.zoom = random.uniform(0.8, zoom)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            
            zoom_image = clipped_zoom(image, self.zoom)
            zoom_label = clipped_zoom(label, self.zoom)

            zoom_image = Image.fromarray(zoom_image.astype('uint8'), 'RGB')
            zoom_label = Image.fromarray(zoom_label.astype('uint8'), 'L')
            return {'image': zoom_image, 'label': zoom_label}

        return {'image': image, 'label': label}


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = scipy.ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = scipy.ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


class Shear(object):
    def __init__(self, shear):
        self.shear = random.uniform(0, shear)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            
            affine_tf = tf.AffineTransform(shear=self.shear)
            shear_image = tf.warp(image, inverse_map=affine_tf)
            shear_label = tf.warp(label, inverse_map=affine_tf)

            shear_image = Image.fromarray(shear_image.astype('uint8'), 'RGB')
            shear_label = Image.fromarray(shear_label.astype('uint8'), 'L')

            return {'image': shear_image, 'label': shear_label}

        return {'image': image, 'label': label}


class Translation(object):
    def __init__(self, translation):
        self.translation = random.uniform(0, translation)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() < 0.5:
            image = np.array(image)
            label = np.array(label)
            rows, cols, ch = image.shape

            tr_x = self.translation/2
            tr_y = self.translation/2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

            translate_image = cv2.warpAffine(image, Trans_M, (cols, rows))
            translate_label = cv2.warpAffine(label, Trans_M, (cols, rows))

            translate_image = Image.fromarray(translate_image.astype('uint8'), 'RGB')
            translate_label = Image.fromarray(translate_label.astype('uint8'), 'L')

            return {'image': translate_image, 'label': translate_label}
        
        return {'image': image, 'label': label}


class RandomCrop(object):

    def __init__(self, output_size=100):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        i, j, h, w = F.get_random_crop_params(image, [self.output_size, self.output_size])
        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        
        return {'image': image, 'label': label}


class Normalization(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


if __name__ == '__main__':

    # transform1 = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip(), RandomZoom(1.2), Shear(0.4), Translation(50), RandomRotation_yq(90), ToTensor()])
    transform1 = transforms.Compose([ToTensor()])

    image_dir = '../train/images/'
    label_dir = '../train/labels/'

    dataset = Dataset_unet(image_dir, label_dir, transform=transform1)
    dataloader = DataLoader(dataset)

    for i_batch, sample_batched in enumerate(dataloader):

        train_image = sample_batched['image']
        print(train_image.shape)

        train_label = sample_batched['label']
        print(train_label.shape)
