from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils import model_zoo

from dataset.DA import DA
import utils.transforms as T
from utils.logging import Logger
from utils.preprocessor import Preprocessor, UnsupervisedCamStylePreprocessor
from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss

def get_data(data_dir, source, target, source_train_path,target_train_path,source_extension,target_extension,height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target,source_train_path,target_train_path,source_extension,target_extension)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    source_num_classes = dataset.num_source_train_ids
    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=True)
    target_train_loader = DataLoader(
        Preprocessor(dataset.target_train, root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=0,
        shuffle=True, pin_memory=False, drop_last=True)
    # source_train_loader = DataLoader(
    #     UnsupervisedCamStylePreprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
    #                                      camstyle_root=osp.join(dataset.source_images_dir, dataset.source_train_path),
    #                  transform=train_transformer),
    #     batch_size=batch_size, num_workers=0,
    #     shuffle=True, pin_memory=False, drop_last=True)
    # target_train_loader = DataLoader(
    #     UnsupervisedCamStylePreprocessor(dataset.target_train,
    #                                      root=osp.join(dataset.target_images_dir, dataset.target_train_path),
    #                                      camstyle_root=osp.join(dataset.target_images_dir,
    #                                                             dataset.target_train_camstyle_path),
    #                                      num_cam=dataset.target_num_cam, transform=train_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return dataset, source_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader

def evaluate(args,model):
    # TODO(tb): implement this code to evaluate a model.
    print('----------------------------- evaluating ----------------------------------')
    print(args)


def train(args,model):
    # TODO(tb): implement this code to train a model
    print('----------------------------- training ----------------------------------')
    print(args)


def fixRandomSeed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    ## fix random_seed
    fixRandomSeed(1)

    ## cuda setting
    cudnn.benchmark = True
    cudnn.enabled = True
    device = torch.device('cuda:' + str(args.gpuid))
    torch.cuda.set_device(device)

    ## Logger setting
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print('logs_dir=', args.logs_dir)
    print('args : ', args)

    ## get dataset & dataloader:
    dataset, source_num_classes, source_train_loader, \
    target_train_loader, query_loader, gallery_loader = get_data(args.data_dir, args.source,args.target,
                                                                 args.source_train_path, args.target_train_path,
                                                                 args.source_extension,args.target_extension,
                                                                 args.height, args.width,
                                                                 args.batch_size, args.re, args.workers)

    h, w = map(int, [args.height,args.width])
    input_size_source = (h, w)
    input_size_target = (h, w)

    # cudnn.enabled = True

    # Create Network
    # model = Res_Deeplab(num_classes=args.num_classes)
    model = Res_Deeplab(num_classes=source_num_classes)
    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()

    ## adapte new_params's layers / classes to saved_state_dict
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    if args.restore_from[:4] == './mo':
        model.load_state_dict(new_params)
    else:
        model.load_state_dict(saved_state_dict)

    ## set mode = train and moves the params of model to GPU
    model.train()
    model.cuda(args.gpu)

    # cudnn.benchmark = True

    # Init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    # =============================================================================
    #    #for retrain
    #    saved_state_dict_D = torch.load(RESTORE_FROM_D)
    #    model_D.load_state_dict(saved_state_dict_D)
    # =============================================================================

    model_D.train()
    model_D.cuda(args.gpu)

    # if not os.path.exists(args.snapshot_dir):
    #     os.makedirs(args.snapshot_dir)

    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                           crop_size=input_size_source,
                           scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=True, mirror=True, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        damping = (1 - i_iter / NUM_STEPS)

        # ======================================================================================
        # train G
        # ======================================================================================

        # Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        pred_source1, pred_source2 = model(images_s)
        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)

        # Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu))
        loss_seg.backward()

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, _, _, _ = batch
        images_t = Variable(images_t).cuda(args.gpu)

        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

        D_out = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

        # Adaptive Adversarial Loss
        if (i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out,
                                         Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                             args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                                Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_adv = loss_adv * Lambda_adv * damping
        loss_adv.backward()

        # Weight Discrepancy Loss
        W5 = None
        W6 = None
        if args.model == 'ResNet':

            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)

        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()

        # ======================================================================================
        # train D
        # ======================================================================================

        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()

        D_out_s = interp_source(model_D(F.softmax(pred_source1 + pred_source2, dim=1)))

        loss_D_s = bce_loss(D_out_s,
                            Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_D_s.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()

        D_out_t = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

        # Adaptive Adversarial Loss
        if (i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t,
                                         Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(
                                             args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t,
                                Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu))

        loss_D_t.backward()

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
                i_iter, args.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))

        f_loss = open(osp.join(args.snapshot_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
            loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))
        f_loss.close()

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))

    ## create dataloader
    dataset, source_num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader = get_data(args.data_dir, args.source,args.target, args.source_train_path, args.target_train_path,
             args.source_extension, args.target_extension, args.height, args.width, args.batch_size, args.re,
             args.workers)
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)
    input_size_target = (h, w)



    # if args.evaluate:
    #     evaluate(args)
    # else:
    #     sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    #     print('logs_dir=', args.logs_dir)
    #     train(args)








if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## system settings
    parser.add_argument('--gpuid', type=str, default='1', help='specify the gpuid ,only one gpu can be specified')
    ## folders settings
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='specify the root path of data, all datasets should be place here')
    parser.add_argument('--log_dir', type=str, default='./logs/market2duke')
    parser.add_argument('--source', type=str, default='market',
                        help='specify the folder/dataset in data_dir,default : market')
    parser.add_argument('--source_train_path', type=str, default='train',
                        help='specify the source_train_path data path in data_dir/source, default : train')
    parser.add_argument('--target', type=str, default='duke',
                        help='specify the target data path in data_dir, default : duke')
    parser.add_argument('--target_train_path', type=str, default='train',
                        help='specify the target_train_path data path in data_dir/target, default : train')
    parser.add_argument('--source_extension', type=str, default='jpg',
                        help='speficy the extension of source images, default : jpg')
    parser.add_argument('--target_extension', type=str, default='jpg',
                        help='speficy the extension of target images, default : jpg')

    # imgs setting
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.5)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for ImageNet pretrained"
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--output_feature', type=str, default='pool5')

    # training configs
    parser.add_argument('--resume', type=str, default='./logs/market2duke/models/latest.pth',
                        help='specify the resume model')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--epochs_decay', type=int, default=40)
    parser.add_argument('--print_freq', type=int, default=30)

    # random erasing
    parser.add_argument('--re', type=float, default=0.5)

    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean')

    # Invariance learning
    parser.add_argument('--inv_alpha', type=float, default=0.01,
                        help='update rate for the exemplar memory in invariance learning')
    parser.add_argument('--inv_beta', type=float, default=0.05,
                        help='The temperature in invariance learning')
    parser.add_argument('--knn', default=6, type=int,
                        help='number of KNN for neighborhood invariance')
    parser.add_argument('--lmd', type=float, default=0.3,
                        help='weight controls the importance of the source loss and the target loss.')

    args = parser.parse_args()
    main(args)