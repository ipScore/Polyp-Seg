# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
from optparse import OptionParser
import torch
from torch import optim
import sys
from unet.unet_model import UNet
from torch.utils.data import DataLoader
import LossFunction_yq
from Dataset_yq_2inputs import Dataset_unet
from torchvision import transforms
from mytransformation_2inputs import ToTensor
from tensorboardX import SummaryWriter


def train_net(image_dir, label_dir, boundary_dir, checkpoint_dir, net, epochs=200, batch_size=4, lr=0.001, save_cp=True, gpu=True):

    print('''
    Starting training: 
        Epochs: {} 
        Batch size: {}
        Learning rate: {}
        Checkpoints: {}
        Want CUDA: {}
    '''.format(epochs, batch_size, lr, str(save_cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 75, 150], gamma=0.1)
    criterion_bce = LossFunction_yq.BCELoss()

    transform1 = transforms.Compose([ToTensor()])

    dataset = Dataset_unet(image_dir, label_dir, boundary_dir, transform=transform1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True)
    dataset_sizes = len(dataset)
    batch_num = int(dataset_sizes/batch_size)

    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        lr_scheduler.step()
        epoch_loss = 0

        for i_batch, sample_batched in enumerate(dataloader):

            #  total_steps = epoch * batch_num + i_batch

            optimizer.zero_grad()
            train_image = sample_batched['image']
            train_boundary = sample_batched['boundary']

            if torch.cuda.is_available() and gpu:
                train_image = train_image.cuda()
                train_boundary = train_boundary.cuda()

            _, pred_bd = net(train_image)
            pred_bd_probs = torch.sigmoid(pred_bd)

            # loss_bd
            loss_bd = criterion_bce(pred_bd_probs, train_boundary)
            # print('loss_bd', loss_bd)

            # total_loss
            loss = loss_bd

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('Epoch finished ! Train Loss (bd): {}'.format(epoch_loss/batch_num))

        writer.add_scalar('Train_Loss (bd)', epoch_loss/batch_num, epoch)

        if save_cp:
            if epoch % 5 == 0:
                torch.save(net.state_dict(), checkpoint_dir + 'CP{}.pth'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-i', '--image_dir', dest='imagedir', default='../train/images/', help='load image directory')
    parser.add_option('-t', '--GT_area_dir', dest='gt', default='../train/labels/', help='load area GT directory')
    parser.add_option('-k', '--GT_boundary_dir', dest='bd', default='../train/boundarytk/', help='load bd GT directory')
    parser.add_option('-p', '--checkpoint_dir', dest='checkpoint', default='../checkpoints/', help='save checkpoint directory')
    parser.add_option('-w', '--tensorboard_dir', dest='tensorboard', default='../train_log', help='save tensorboard directory')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    args = get_args()
    print(args)

    have_gpu = torch.cuda.is_available()
    print('Have GPU?:{}'.format(have_gpu))

    writer = SummaryWriter(args.tensorboard)

    # --------------------------- using pre-trained params ---------------------------------- #

    # (1) get param from pre-trained model
    # from unet_3up_area.unet.unet_model import UNet as UNet_old
    from unet_3up_area.unet_model import UNet as UNet_old
    net_old = UNet_old(n_channels=3, n_classes=1)
    net_old.load_state_dict(torch.load('../load_model_from_step1_area_branch_with_sknet/CPxx.pth'))
    net_old_dict = net_old.state_dict()

    # (2) our new model
    net = UNet(n_channels=3, n_classes=1)
    net_dict = net.state_dict()

    # # (3) apply pre-trained params in new model
    net_old_dict = {k: v for k, v in net_old_dict.items() if k in net_dict}
    net_dict.update(net_old_dict)  # update params using pre-trained model
    net.load_state_dict(net_dict)  # update the model

    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # -------------------------- fix parameters related area ---------------------------------- #

    # whether fix parameters related area
    for name, param in net.named_parameters():
        if "down" in name:
            param.requires_grad = False

        if "inc" in name:
            param.requires_grad = False

    if have_gpu and args.gpu:
        print('Using GPU !')
        net = net.cuda()

    try:
        train_net(image_dir=args.imagedir,
                  label_dir=args.gt,
                  boundary_dir=args.bd,
                  checkpoint_dir=args.checkpoint,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu
                  )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

