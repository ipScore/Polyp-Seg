from __future__ import division
from PIL import Image
import torch
from unet.unet_model import UNet
from Dataset_yq_2inputs import Dataset_unet
from torch.utils.data import DataLoader
from torchvision import transforms
from mytransformation_2inputs import ToTensor
import os
from os.path import *
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


batch_size = 1
test_image_dir = '../test/images/'
test_label_dir = '../test/labels/'
test_boundary_dir = '../test/boundarytk/'
checkpoints_dir = '../checkpoints/'

#save_path = 'test_results/'
#if not exists(save_path):
#    os.mkdir(save_path)

net = UNet(n_channels=3, n_classes=1)
net.cuda()
net.eval()

for checkpoint in range(1, 41):
    net.load_state_dict(torch.load(checkpoints_dir + 'CP' + str(5 * checkpoint - 4) + '.pth'))

    transform1 = transforms.Compose([ToTensor()])
    test_dataset = Dataset_unet(test_image_dir, test_label_dir, test_boundary_dir, transform=transform1)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    dataset_sizes = len(test_dataset)
    batch_num = int(dataset_sizes / batch_size)

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
        test_image = sample_batched['image']
        test_label = sample_batched['label']
        test_idx = sample_batched['num'][0]
        # print(train_idx)

        if torch.cuda.is_available():
            test_image = test_image.cuda()
            test_label = test_label.cuda()

        predict_label, _, _ = net(test_image)
        predict_probs = torch.sigmoid(predict_label)

        ################################## save the results ################################

        #predict_probs_255 = torch.squeeze(255 * predict_probs)
        #predict_probs_255 = predict_probs_255.cpu().detach().numpy()  # size is (288, 384)
        
        #predict_final = Image.fromarray(predict_probs_255)
        #predict_final = predict_final.convert("L")
        #predict_final.save(os.path.join(save_path, test_idx))

        ########################## print the measurement metrics ############################

        predict_probs_rep = predict_probs
        predict_probs_rep = (predict_probs_rep >= 0.5).float()
        test_label_rep = test_label
        test_label_rep = (test_label_rep >= 0.5).float()

        label_probs_rep_inverse = predict_probs_rep
        label_probs_rep_inverse = (label_probs_rep_inverse == 0).float()

        train_label_rep_inverse = test_label_rep
        train_label_rep_inverse = (train_label_rep_inverse == 0).float()

        TP = predict_probs_rep.mul(test_label_rep).sum()
        FP = predict_probs_rep.mul(train_label_rep_inverse).sum()
        TN = label_probs_rep_inverse.mul(train_label_rep_inverse).sum()
        FN = label_probs_rep_inverse.mul(test_label_rep).sum()

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

    f = open("/home/zhudelong/project/yuqi/unet_3up_ab_indrt_fixa/3ab_fixa_cst_local/1234/test_results_1234.txt", "a")
    print >> f, "%0.16f" % (F1 / batch_num)
    f.close()
