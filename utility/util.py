import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import random
import utility.load_wordembeddings as load_wordembeddings
from torch.utils.data import Dataset

from torch import Tensor
import torch.nn as nn

class cos_sim_loss(nn.MSELoss):
    __constants__ = ['reduction']
    def __init__(self, dim=1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(cos_sim_loss, self).__init__(size_average, reduce, reduction)
        assert reduction in ['none', None, 'mean', 'sum']
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=dim, eps=1e-8)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = 1-self.cos(input, target) 
        if self.reduction == 'none' or self.reduction == None:
            return loss.unsqueeze(-1)
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum() 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

# Maps test classes to 151-200 instead of 1-50 (for latter, use map_label)
def map_label_extend(label, new_classes, base_classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(new_classes.size(0)):
        mapped_label[label==new_classes[i]] = i + len(base_classes)
    return mapped_label

def reverse_map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==i] = classes[i]    
    return mapped_label

def reverse_map_label_extend(label, new_classes, base_classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(new_classes.size(0)):
        mapped_label[label==(i + len(base_classes))] = new_classes[i]
    return mapped_label

class ClsWeightsDataset(Dataset):
    def __init__(self, opt, attributes, target_weights, seenclasses, cuda, transform=None):
        assert len(attributes[seenclasses]) == len(target_weights)
        self.opt = opt
        self.attributes = attributes
        self.target = target_weights
        self.seenclasses = seenclasses
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.seenclasses)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        att = self.attributes[self.seenclasses[idx]]
        target = self.target[idx]
            
        if self.cuda:
            att = att.cuda()
            target = target.cuda()

        if self.transform:
            att = self.transform(att)
        
        return att, target
        

class GenericDataset(Dataset):
    def __init__(self, opt, _input, _target, cuda, transform=None):
        assert len(_input) == len(_target)
        self.opt = opt
        self.input = _input
        self.target = _target
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        in_var = self.input[idx]
        target = self.target[idx]

        if self.cuda:
            in_var = in_var.cuda()
            target = target.cuda()
    
        if self.transform:
            in_var = self.transform(in_var)
        
        return in_var, target


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
    
    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        
        feature = matcontent['features'].T
        if opt.image_embedding[:10] in ['pretrained']: # Embedding from Massi's feature extractor code (shape transposed compared to Yongqin features)
            feature = feature.T
        label = matcontent['labels'].astype(int).squeeze() - 1

        # Change class embedding (attributes or label embeddings)
        if opt.class_embedding in ['wiki2vec', 'cn', 'clip']:
            mat_path = opt.dataroot + "/" + opt.dataset + "/" + 'att' + "_splits.mat"
        else:
            mat_path = opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat"
        matcontent = sio.loadmat(mat_path)

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        
        if opt.zst:
            if opt.class_embedding == 'wiki2vec':
                transfer_path = opt.rootpath + "/embeddings/wiki2vec/" + opt.dataset + "_wiki_sum_list.npy"
                transfer_attributes = torch.from_numpy(np.load(transfer_path, allow_pickle=True)).float()
                transfer_attributes /= torch.norm(transfer_attributes, dim=1)[:, None]
                if opt.zstfrom == 'imagenet':
                    source_path = opt.rootpath + "/embeddings/wiki2vec/imgnet_wiki_list.npy"
                else:
                    source_path = opt.rootpath + "/embeddings/wiki2vec/" + opt.zstfrom + "_wiki_sum_list.npy"
                source_attributes = torch.from_numpy(np.load(source_path, allow_pickle=True)).float()
                source_attributes /= torch.norm(source_attributes, dim=1)[:, None]
            
            elif opt.class_embedding == 'cn':
                transfer_path = opt.rootpath + "/embeddings/conceptnet/" + opt.dataset + "_cn_sum_list.npy"
                transfer_attributes = torch.from_numpy(np.load(transfer_path, allow_pickle=True)).float()
                transfer_attributes /= torch.norm(transfer_attributes, dim=1)[:, None]
                
                if opt.zstfrom == 'imagenet':
                    source_path = opt.rootpath + "/embeddings/conceptnet/imgnet_cn_list.npy"
                else:
                    source_path = opt.rootpath + "/embeddings/conceptnet/" + opt.zstfrom + "_cn_sum_list.npy"
                source_attributes = torch.from_numpy(np.load(source_path, allow_pickle=True)).float()
                source_attributes /= torch.norm(source_attributes, dim=1)[:, None]
                
            elif opt.class_embedding == 'clip':
                vocab = load_wordembeddings.prepare_vocab(opt, matcontent)
                transfer_attributes = load_wordembeddings.get_clip_embeddings(opt, vocab)
                if opt.zstfrom == 'imagenet':
                    with open(opt.rootpath + "/utility/ImageNet1K_classnames.txt") as f:
                        in1k_classnames = f.read().splitlines()
                    source_classnames = load_wordembeddings.prep_imagenet_vocab(in1k_classnames)
                else:
                    sourcematcontent = sio.loadmat(opt.dataroot + "/" + opt.zstfrom + "/" + opt.image_embedding + ".mat")
                    sourcelabel = sourcematcontent['labels'].astype(int).squeeze() - 1
                    source_path = opt.dataroot + "/" + opt.zstfrom + "/" + 'att' + "_splits.mat"
                    sourcematcontent = sio.loadmat(source_path)

                    source_seen_loc = sourcematcontent['test_seen_loc'].squeeze() - 1
                    source_seenclasses = np.unique(torch.from_numpy(sourcelabel[source_seen_loc]).long().numpy())

                    source_classnames = load_wordembeddings.prepare_vocab(opt, sourcematcontent, zst_mode=True)
                    source_classnames = np.array(source_classnames)[source_seenclasses].tolist()
                    
                source_attributes = load_wordembeddings.get_clip_embeddings(opt, source_classnames)

            else:
                raise NotImplementedError
            
            self.attribute = torch.cat((source_attributes, transfer_attributes)) 
 
        else:
            if opt.class_embedding == 'wiki2vec':
                embedding_path = opt.rootpath + "/embeddings/wiki2vec/" + opt.dataset + "_wiki_sum_list.npy"
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                print("attributes pre normalization", self.attribute, self.attribute.shape)
                print(torch.norm(self.attribute, dim=1), torch.norm(self.attribute, dim=1).shape)
                self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
                print("attributes post normalization", self.attribute, self.attribute.shape)
            elif opt.class_embedding == 'cn':
                embedding_path = opt.rootpath + "/embeddings/conceptnet/" + opt.dataset + "_cn_sum_list.npy"
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
            elif opt.class_embedding == 'clip':
                vocab = load_wordembeddings.prepare_vocab(opt, matcontent)
                self.attribute = load_wordembeddings.get_clip_embeddings(opt, vocab)
                print(self.attribute.shape)
            else:
                assert opt.class_embedding in ['att']
                self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        
        print("Loaded attributes / word embeddings with shape", self.attribute.shape)
        
        if opt.preprocessing:
            if opt.standardization:
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()
            
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1/mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1/mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
            self.test_seen_feature.mul_(1/mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
            self.train_label = torch.from_numpy(label[trainval_loc]).long() 
            self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
            self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        if opt.zst:
            self.unseenclasses = self.unseenclasses + len(source_attributes)
            self.seenclasses = torch.arange(len(source_attributes))

        self.ntrain = self.train_feature.size()[0]
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 
