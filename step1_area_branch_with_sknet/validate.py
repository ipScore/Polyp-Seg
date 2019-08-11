from __future__ import division
import torch
from optparse import OptionParser
from unet.unet_model import UNet
from Dataset_yq_2inputs import Dataset_unet
from torch.utils.data import DataLoader
from torchvision import transforms
from mytransformation_2inputs import ToTensor
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
import re
import LossFunction_yq
import torchvision.transforms.functional as F


def predict(validate_image_dir, validate_label_dir, checkpoints_dir, net, batch_size=1, gpu=True):
    transform1 = transforms.Compose([ToTensor()])
    file_num = 31
    validate_dataset = Dataset_unet(validate_image_dir, validate_label_dir, transform=transform1)
    dataloader = DataLoader(validate_dataset, batch_size=batch_size)
    dataset_sizes = len(validate_dataset)
    batch_num = int(dataset_sizes/batch_size)

    for epoch in range(1, file_num):
        net.load_state_dict(torch.load(checkpoints_dir + 'CP' + str(5*epoch-4) + '.pth'))

        Sensitivity = 0
        Specificity = 0
        Precision = 0
        F1 = 0
        F2 = 0
        ACC_overall = 0
        IoU_poly = 0
        IoU_bg = 0
        IoU_mean = 0

        for i_batch, sample_batched in enumerate(dataloader):
            validate_image = sample_batched['image']
            validate_label = sample_batched['label']

            if torch.cuda.is_available() and gpu:
                validate_image = validate_image.cuda()
                validate_label = validate_label.cuda()

            predict_label = net(validate_image)
            predict_probs = torch.sigmoid(predict_label)

            predict_probs_rep = predict_probs
            predict_probs_rep = (predict_probs_rep >= 0.5).float()
            #
            validate_label_rep = validate_label
            validate_label_rep = (validate_label_rep >= 0.5).float()

            label_probs_rep_inverse = predict_probs_rep
            label_probs_rep_inverse = (label_probs_rep_inverse == 0).float()

            train_label_rep_inverse = validate_label_rep
            train_label_rep_inverse = (train_label_rep_inverse == 0).float()

            # calculate TP, FP, TN, FN
            TP = predict_probs_rep.mul(validate_label_rep).sum()
            FP = predict_probs_rep.mul(train_label_rep_inverse).sum()
            TN = label_probs_rep_inverse.mul(train_label_rep_inverse).sum()
            FN = label_probs_rep_inverse.mul(validate_label_rep).sum()

            if TP.item() == 0:
                # print('TP=0 now!')
                # print('Epoch: {}'.format(epoch))
                # print('i_batch: {}'.format(i_batch))

                TP = torch.Tensor([1]).cuda()

            # Sensitivity, hit rate, recall, or true positive rate
            temp_Sensitivity = TP / (TP + FN)

            # Specificity or true negative rate
            temp_Specificity = TN / (TN + FP)

            # Precision or positive predictive value
            temp_Precision = TP / (TP + FP)

            # F1 score = Dice
            temp_F1 = 2 * temp_Precision * temp_Sensitivity / (temp_Precision + temp_Sensitivity)

            # F2 score
            temp_F2 = 5 * temp_Precision * temp_Sensitivity / (4 * temp_Precision + temp_Sensitivity)

            # Overall accuracy
            temp_ACC_overall = (TP + TN) / (TP + FP + FN + TN)

            # Mean accuracy
            # temp_ACC_mean = TP / pixels

            # IoU for poly
            temp_IoU_poly = TP / (TP + FP + FN)

            # IoU for background
            temp_IoU_bg = TN / (TN + FP + FN)

            # mean IoU
            temp_IoU_mean = (temp_IoU_poly + temp_IoU_bg) / 2.0

            # To Sum
            Sensitivity += temp_Sensitivity.item()
            Specificity += temp_Specificity.item()
            Precision += temp_Precision.item()
            F1 += temp_F1.item()
            F2 += temp_F2.item()
            ACC_overall += temp_ACC_overall.item()
            IoU_poly += temp_IoU_poly.item()
            IoU_bg += temp_IoU_bg.item()
            IoU_mean += temp_IoU_mean.item()

        writer.add_scalar('Validate/sensitivity', Sensitivity / batch_num, epoch)
        writer.add_scalar('Validate/specificity', Specificity / batch_num, epoch)
        writer.add_scalar('Validate/precision', Precision / batch_num, epoch)
        writer.add_scalar('Validate/F1', F1 / batch_num, epoch)
        writer.add_scalar('Validate/F2', F2 / batch_num, epoch)
        writer.add_scalar('Validate/ACC_overall', ACC_overall / batch_num, epoch)
        writer.add_scalar('Validate/IoU_poly', IoU_poly / batch_num, epoch)
        writer.add_scalar('Validate/IoU_bg', IoU_bg / batch_num, epoch)
        writer.add_scalar('Validate/IoU_mean', IoU_mean / batch_num, epoch)



def get_args():
    parser = OptionParser()
    parser.add_option('-i', '--validate_image_dir', dest='imagedir', default='../validate/images/', help='load validation image directory')
    parser.add_option('-t', '--validate_label_dir', dest='gt', default='../validate/labels/', help='load validation area GT directory')
    parser.add_option('-p', '--checkpoint_dir', dest='checkpoint', default='../checkpoints/', help='save checkpoint directory')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1, type='int', help='batch size')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-w', '--tensorboard_dir', dest='tensorboard', default='../validate_log', help='save tensorboard directory')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    args = get_args()
    print(args)

    writer = SummaryWriter(args.tensorboard)

    have_gpu = torch.cuda.is_available()
    print('Have GPU?:{}'.format(have_gpu))

    net = UNet(n_channels=3, n_classes=1)
    net.eval()

    if have_gpu and args.gpu:
        net = net.cuda()
        print('Using GPU !')

    predict(validate_image_dir=args.imagedir,
            validate_label_dir=args.gt,
            checkpoints_dir=args.checkpoint,
            net=net,
            batch_size=args.batchsize,
            gpu=args.gpu
            )

    ## tensorboard --logdir=./log* --port=8008


