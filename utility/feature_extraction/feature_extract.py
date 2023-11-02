from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
from utility.feature_extraction.extract_util import get_loader, prepare_attri_label, DATA_LOADER, map_label
from opt import get_opt
import torchvision
import timm
import scipy

cudnn.benchmark = True

opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)



def main():
    # load data
    data = DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label  # weird

    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()


    # define test_classes
    if opt.image_type not in ['test_unseen_small_loc', 'test_unseen_loc', 'test_seen_loc']:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")


    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # define attribute groups
    if opt.dataset == 'CUB':
        # Change layer
        num_classes = 150
    elif opt.dataset == 'AWA2':
        # Change layer
        num_classes = 40
    elif opt.dataset == 'SUN':
        # Change layer
        num_classes = 645

    if 'vit' in opt.backbone:
        model = timm.create_model(opt.backbone,pretrained=True,num_classes=num_classes)
        if opt.save_features:
            model.head = nn.Identity()
    else:
        ####### load our network, any from here: https://pytorch.org/vision/0.11/models #######
        if opt.backbone == 'resnet101_old':
            model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            if opt.resnet_path is not None:
                model = torchvision.models.__dict__[opt.backbone](pretrained=False)
                model.load_state_dict(torch.load(opt.resnet_path))
            else:
                model = torchvision.models.__dict__[opt.backbone](pretrained=True)
        model.fc = nn.Linear(opt.feature_size, num_classes)
        if opt.save_features:
            model.fc = nn.Identity()
    print(model)



    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()

    if opt.save_features:
        name=opt.dataset+'_'+opt.backbone+'_fix'+'.mat'
        model.eval()
        img_files = []
        features = []
        labels = []
        with torch.no_grad():
            loaders = [trainloader]
            for loader in loaders:
                for i, (batch_input, batch_target, impath) in enumerate(loader):
                    input_v = Variable(batch_input)
                    if opt.cuda:
                        input_v = input_v.cuda()
                    output = model(input_v).to('cpu')
                    for j in range(len(batch_target)):
                        img_files.append([np.array([impath[j].squeeze().replace(' ','')])])
                        labels.append(np.array([batch_target[j].item()+1],dtype=np.int16))
                        features.append(output[j].numpy())
            scipy.io.savemat(name, mdict={'image_files': img_files, 'features': features, 'labels': np.array(labels)})

        exit(0)


    print('Train and test...')
    for epoch in range(opt.nepoch):
        model.train()
        current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=current_lr, betas=(opt.beta1, 0.999))
        # loss for print
        loss_log = {'ave_loss': 0}

        batch = len(trainloader)
        for i, (batch_input, batch_target, impath) in enumerate(trainloader):
            model.zero_grad()
            # map target labels
            batch_target = map_label(batch_target, data.seenclasses)
            input_v = Variable(batch_input)
            label_v = Variable(batch_target)
            if opt.cuda:
                input_v = input_v.cuda()
                label_v = label_v.cuda()
            output = model(input_v)

            loss = criterion(output, label_v)
            loss_log['ave_loss'] += loss.item()
            loss.backward()
            optimizer.step()

        print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
              % (epoch+1, batch, loss_log['ave_loss'] / batch))



if __name__ == '__main__':
    main()