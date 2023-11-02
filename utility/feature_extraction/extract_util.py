from sklearn import preprocessing
import sys
import torch.utils.data
import os
from PIL import Image
import numpy as np
import h5py
import torch
import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms

def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        self.image_files = matcontent['image_files'].squeeze()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        if opt.dataset == 'CUB':
            self.train_loc = matcontent['train_loc'].squeeze() - 1
            self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1

        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.allclasses_name = matcontent['allclasses_names']
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[self.trainval_loc])
                _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[self.trainval_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[self.test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[self.test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
            self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[self.val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
     
        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att

def prepare_attri_label(attribute, classes):
    classes_dim = classes.size(0)
    attri_dim = attribute.shape[1]
    output_attribute = torch.FloatTensor(classes_dim, attri_dim)
    for i in range(classes_dim):
        output_attribute[i] = attribute[classes[i]]
    return torch.transpose(output_attribute, 1, 0)

def get_loader(opt, data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if opt.transform_complex:
        train_transform = []
        size = 224
        train_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        train_transform = transforms.Compose(train_transform)
        test_transform = []
        size = 224
        test_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose(test_transform)
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if opt.save_features:
        dataset_train = ImageFilelist(opt, data_inf=data,
                                      transform=train_transform,
                                      dataset=opt.dataset,
                                      image_type='all')
    else:
        dataset_train = ImageFilelist(opt, data_inf=data,
                                      transform=train_transform,
                                      dataset=opt.dataset,
                                      image_type='trainval_loc')

    if opt.save_features:
        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

    if not opt.save_features:
        dataset_test_unseen = ImageFilelist(opt, data_inf=data,
                                            transform=test_transform,
                                            dataset=opt.dataset,
                                            image_type='test_unseen_loc')
        testloader_unseen = torch.utils.data.DataLoader(
            dataset_test_unseen,
            batch_size=opt.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        dataset_test_seen = ImageFilelist(opt, data_inf=data,
                                          transform=transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]),
                                          dataset=opt.dataset,
                                          image_type='test_seen_loc')
        testloader_seen = torch.utils.data.DataLoader(
            dataset_test_seen,
            batch_size=opt.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        # dataset for visualization (CenterCrop)
        dataset_visual = ImageFilelist(opt, data_inf=data,
                                       transform=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]),
                                       dataset=opt.dataset,
                                       image_type=opt.image_type)

        visloader = torch.utils.data.DataLoader(
            dataset_visual,
            batch_size=opt.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        return trainloader, testloader_unseen, testloader_seen, visloader
    else:
        return trainloader, None,None,None