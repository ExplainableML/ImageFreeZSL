import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import math
import itertools
import utility.util as util
import sys
import joint_latent
import datetime 

from utility.train_base import BASECLASSIFIER
import baselines.conse as conse
import baselines.vgse as vgse
import baselines.costa as costa

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help='Enables cuda')
parser.add_argument('--dataroot', default='/home/andchri/Feature-Generation-datasets', help='Path to datasets folder')
parser.add_argument('--rootpath', default='/home/andchri/APZSL-pub', help='Path for saving model checkpoints and results')
parser.add_argument('--numSeeds', type=int, default=1, help='Number of (randomly selected) seeds to experiment on')
parser.add_argument('--manualSeed', nargs='+', type=int, default=None, help='Specify manual seed(s)')

parser.add_argument('--dataset', default='CUB', help='Dataset for zero-shot classification')
parser.add_argument('--image_embedding', default='res101_finetuned', help='Whether base classifier was finetuned on seen classes or generic imagenet model')
parser.add_argument('--class_embedding', default='att', help='Semantic class-wise information')
parser.add_argument('--zst', action='store_true', default=False, help='Perform experiment of model transfer from one dataset to another')
parser.add_argument('--zstfrom', default='imagenet', help='Transfer from which dataset [imagenet, cub, sun, awa2]')

parser.add_argument('--strict_eval', action='store_true', default=False, help='When running on test set, only validate after final epoch')
parser.add_argument('--save_pred_matrix', action='store_true', default=False, help='Save matrices with predictions after evaluation')
parser.add_argument('--early_stopping_slope', action='store_true', default=False, help='Enable early stopping heuristic')
parser.add_argument('--cos_sim_loss', action='store_true', default=False, help='Enable cosine similarity loss')
parser.add_argument('--include_unseen', action='store_true', default=False, help='Whether to include unseen attributes during training')
parser.add_argument('--norm_scale_heuristic', action='store_true', default=False, help='Scale the predicted classifier weights (heuristic for bias correction)')

# Training args
parser.add_argument('--num_layers', nargs='+', type=int, default=[2], help='Number of layers in weight prediction MLP (2, 3, or 4)')
parser.add_argument('--embed_dim', nargs='+', type=int, default=[1000], help='Set the dimensionality of the hidden layers')
parser.add_argument('--batch_size', nargs='+', type=int, default=[16], help='input batch size')
parser.add_argument('--nepoch', nargs='+', type=int, default=[1000], help='Max number of epochs to train for')
parser.add_argument('--classifier_nepoch', type=int, default=100, help='Max number of epochs to train for')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='Learning rate to train softmax classifier')
parser.add_argument('--classifier_beta1', type=float, default=0.9, help='beta1 for adam to train classifier. default=0.5') # can be removed?
parser.add_argument('--lr', nargs='+', type=float, default=[0.0001], help='Learning rate(s) to train weight regressor network')
parser.add_argument('--beta1', nargs='+', type=float, default=[0.9], help='beta1 parameter(s) for adam to train weight regressor network. default=0.5')

# Baselines
parser.add_argument('--conse_benchmark', action='store_true', default=False, help='Run ConSE benchmark')
parser.add_argument('--costa_benchmark', action='store_true', default=False, help='Run COSTA benchmark')
parser.add_argument('--subspace_proj', action='store_true', default=False, help='Adapted baseline from Akyürek et al. Project predicted weights unto subspace spanned by seen class weights')
parser.add_argument('--vgse_baseline', default=None, help='Run VGSE CRM baseline (choices: wavg or smo)')
parser.add_argument('--vgse_nbs', default=5, help='Number of VGSE CRM WAvg neighbours')
parser.add_argument('--vgse_eta', default=5, help='eta hyperparameter for VGSE CRM WAvg')
parser.add_argument('--vgse_alpha', type=float, default=0, help='alpha hyperparameter for VGSE CRM SMO')
parser.add_argument('--daegnn', action='store_true', default=False, help='Run wDAE-GNN benchmark')

# Ablation args
parser.add_argument('--single_autoencoder_baseline', action='store_true', default=False, help='Train a single autoencoder predicting weights from attributes')
parser.add_argument('--single_modal_ablation', action='store_true', default=False, help='Ablation: remove Weight to Attribute mapping')
parser.add_argument('--class_reduction_ablation', type=int, default=0, help='Run ablation with reducing number of seen classes (0 = No ablation)')
parser.add_argument('--calc_entropy', action='store_true', default=False, help='Calculate output distribution on test set of seen and unseen classes')

# ZSL dataloader args
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--preprocessing', action='store_true', default=False, help='Enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)

opt = parser.parse_args()

assert opt.image_embedding in ['res101_finetuned', 'pretrained_resnet101'], "Available image embeddings."
assert opt.class_embedding in ['att', 'wiki2vec', 'cn', 'clip'], "Available class embeddings are att (attributes), wiki2vec, cn (ConceptNet) and clip (CLIP embeddings)."
if opt.save_pred_matrix:
    assert opt.strict_eval, "If saving prediction matrices, run with strict_eval to not overwrite"
if opt.vgse_baseline:
    assert opt.vgse_baseline in ['wavg', 'smo']
if opt.zst:
    assert opt.zstfrom in ['imagenet', 'CUB', 'SUN', 'AWA2']
    if opt.zstfrom == 'imagenet':
       assert opt.image_embedding == 'pretrained_resnet101', "Use basic pretrained res101 features when doing ZST from imagenet"

if not os.path.exists(opt.rootpath):
    os.makedirs(opt.rootpath)

if opt.manualSeed is None:
    seedlist = [random.randint(1, 10000) for _ in range(opt.numSeeds)]
else:
    opt.numSeeds = len(opt.manualSeed)
    seedlist = opt.manualSeed

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

args = [opt.lr, opt.beta1, opt.nepoch, opt.batch_size, opt.embed_dim, opt.num_layers]
params = [x if type(x)==list else [x] for x in args]
params = list(itertools.product(*params))
print(f"Running experiment on {opt.dataset} with using {opt.class_embedding} as class descriptors.")

accs_unseen_only, accs_gzsl, accs_unseen, accs_seen, hs = [], [], [], [], []
accs_unseen_only_std, accs_gzsl_std, accs_unseen_std, accs_seen_std, hs_std = [], [], [], [], []
hparam_avg_mses, hparams_min_mses, epoch_min_idx_argmax, epoch_min_idx_mean, hparam_mse_idxs, hparam_loss_idxs, hparams_min_losses, hparam_cos_idxs, hparams_max_cos, hparam_avg_cos_list = [], [], [], [], [], [], [], [], [], []
start_time = datetime.datetime.now()
for _lr, _beta1, _nepoch, _batch_size, _embed_dim, _num_layers in params:
    acc_gzsl_seeds_avg, acc_seen_seeds_avg, acc_unseen_seeds_avg, H_seeds_avg, unseen_zsl_seeds_avg = [], [], [], [], []
    seed_mse_list, seed_min_mse, seed_min_idx, seed_mse_idx, seed_min_loss, seed_loss_idx, seed_cos_idx, seed_max_cos, seed_avg_cos_list = [], [], [], [], [], [], [], [], []

    for seed in seedlist:
        split_mse_list, split_min_idx_list, split_loss_list, split_loss_idx_list, split_cos_list, split_cos_idx_list, split_cos_full_list  = [], [], [], [], [], [], []
        print("Random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if opt.cuda:
            torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

        # load data
        data = util.DATA_LOADER(opt)
        nclass = len(data.seenclasses) + len(data.unseenclasses)
        
        # load or train base classification model
        if not opt.zst:
            if not os.path.exists(opt.rootpath + '/models/base-classifiers/'):
                os.makedirs(opt.rootpath + '/models/base-classifiers/')
            model_path = opt.rootpath + '/models/base-classifiers/' + opt.dataset + opt.image_embedding + f'_seed{seed}_clr{opt.classifier_lr}_nep{opt.classifier_nepoch}'
            if os.path.isfile(model_path):
                print(f"Existing base classifier for dataset {opt.dataset} on seed {seed} with given classifier training settings detected. Loading model and skipping training.")
            else:
                base_model = BASECLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data, nclass, opt.cuda, seedinfo=seed,
                                    _lr=_lr, _beta1=_beta1, _nepoch=opt.classifier_nepoch, _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt).fit()
                torch.save(base_model, model_path)
                print(f"Saved base classifier for dataset {opt.dataset} trained on seed {seed}.")
            
        if opt.vgse_baseline:
            VGSE = vgse.VGSE_CRM(_train_X=data.train_feature, _train_Y=util.map_label(data.train_label, data.seenclasses), data_loader=data, _nclass=nclass, _cuda=opt.cuda, seedinfo=seed,
                                _lr=_lr, _beta1=_beta1, _nepoch=_nepoch, _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt)
            if opt.zst:        
                acc_unseen_only, acc_unseen = VGSE.acc_target, VGSE.acc_zst_unseen
            else:
                acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_only = VGSE.acc_gzsl, VGSE.acc_seen, VGSE.acc_unseen, VGSE.H, VGSE.acc_unseen_zsl
        elif opt.conse_benchmark:
            bs = opt.batch_size[0]
            ConSE = conse.ConSE(_train_X=data.train_feature, _train_Y=util.map_label(data.train_label, data.seenclasses), data_loader=data, _nclass=nclass, _cuda=opt.cuda, seedinfo=seed,
                                _lr=_lr, _beta1=_beta1, _nepoch=_nepoch, _batch_size=bs, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt)
            if opt.zst:
                acc_unseen_only, acc_unseen = ConSE.acc_target, ConSE.acc_zst_unseen
            else:
                acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_only = ConSE.acc_gzsl, ConSE.acc_seen, ConSE.acc_unseen, ConSE.H, ConSE.acc_unseen_zsl    
        elif opt.costa_benchmark:
            COSTA = costa.COSTA(_train_X=data.train_feature, _train_Y=util.map_label(data.train_label, data.seenclasses), data_loader=data, _nclass=nclass, _cuda=opt.cuda, seedinfo=seed,
                             _lr=_lr, _beta1=_beta1, _nepoch=_nepoch, _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt) 
            if opt.zst:
                acc_unseen_only, acc_unseen = COSTA.acc_target, COSTA.acc_zst_unseen
            else:
                acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_only = COSTA.acc_gzsl, COSTA.acc_seen, COSTA.acc_unseen, COSTA.H, COSTA.acc_unseen_zsl    
        else:
            MODEL = joint_latent.Joint(data.train_feature, util.map_label(data.train_label, data.seenclasses), data, nclass, opt.cuda, seedinfo=seed,
                            _lr=_lr, _beta1=_beta1, _nepoch=_nepoch, _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt) 
            MODEL.fit()
            if opt.zst:
                acc_unseen_only, acc_unseen = MODEL.acc_target, MODEL.acc_zst_unseen
            else:
                acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_only = MODEL.acc_gzsl, MODEL.acc_seen, MODEL.acc_unseen, MODEL.H, MODEL.acc_unseen_zsl    

        if opt.zst:
            print(f"I-ZSL accuracy from {opt.zstfrom} transfer: {acc_unseen_only*100:.2f}%.")
            print(f"Unseen accuracy (not H) I-GZSL from {opt.zstfrom} transfer: {acc_unseen*100:.2f}%.")
        else:
            print(f"I-ZSL (unseen only) Acc = {acc_unseen_only*100:.2f}%")
            print(f"I-GZSL (seen and unseen): H = {H*100:.2f}, Seen={acc_seen*100:.2f}%, Unseen={acc_unseen*100:.2f}%")
        print("-------")
    
        unseen_zsl_seeds_avg.append(acc_unseen_only)
        acc_unseen_seeds_avg.append(acc_unseen)
        if not opt.zst:
            acc_gzsl_seeds_avg.append(acc_gzsl)
            acc_seen_seeds_avg.append(acc_seen)
            H_seeds_avg.append(H)
    
    accs_unseen_only.append(torch.std_mean(torch.stack(unseen_zsl_seeds_avg), dim=0, unbiased=False)[1])
    accs_unseen_only_std.append(torch.std_mean(torch.stack(unseen_zsl_seeds_avg), dim=0, unbiased=False)[0])
    accs_unseen.append(torch.std_mean(torch.stack(acc_unseen_seeds_avg), dim=0, unbiased=False)[1])
    accs_unseen_std.append(torch.std_mean(torch.stack(acc_unseen_seeds_avg), dim=0, unbiased=False)[0])
    if not opt.zst:
        accs_gzsl.append(torch.std_mean(torch.stack(acc_gzsl_seeds_avg), dim=0, unbiased=False)[1])
        accs_gzsl_std.append(torch.std_mean(torch.stack(acc_gzsl_seeds_avg), dim=0, unbiased=False)[0])
        accs_seen.append(torch.std_mean(torch.stack(acc_seen_seeds_avg), dim=0, unbiased=False)[1])
        accs_seen_std.append(torch.std_mean(torch.stack(acc_seen_seeds_avg), dim=0, unbiased=False)[0])
        hs.append(torch.std_mean(torch.stack(H_seeds_avg), dim=0, unbiased=False)[1])
        hs_std.append(torch.std_mean(torch.stack(H_seeds_avg), dim=0, unbiased=False)[0])

accs_unseen_only = torch.stack(accs_unseen_only)
accs_unseen_only_std = torch.stack(accs_unseen_only_std)
accs_unseen = torch.stack(accs_unseen)
accs_unseen_std = torch.stack(accs_unseen_std)
idx_best_unseen = torch.argmax(accs_unseen)
if not opt.zst:
    accs_gzsl = torch.stack(accs_gzsl)
    accs_gzsl_std = torch.stack(accs_gzsl_std)
    accs_seen = torch.stack(accs_seen)
    accs_seen_std = torch.stack(accs_seen_std)
    hs = torch.stack(hs)
    hs_std = torch.stack(hs_std)

    idx_best_H = torch.argmax(hs)

if opt.numSeeds > 1:
    if opt.zst:
        print(f"Performance, meaned over seeds: I-ZSL (unseen only) Acc = ({accs_unseen_only[idx_best_unseen]*100:.2f} +/- {accs_unseen_only_std[idx_best_unseen]*100:.2f})%, I-GZSL Unseen acc  \
        = ({accs_unseen[idx_best_H]*100:.2f} +/- {accs_unseen_std[idx_best_H]*100:.2f})%, averaged over seeds {opt.manualSeed}. For Seen accuracy (and thus H), evaluate using eval_imagenet.py")
    else:
        print(f"Performance, meaned over seeds: I-ZSL (unseen only) Acc = ({accs_unseen_only[idx_best_H]*100:.2f} +/- {accs_unseen_only_std[idx_best_H]*100:.2f})%, I-GZSL (seen and unseen) H = ({hs[idx_best_H]*100:.2f} +/- {hs_std[idx_best_H]*100:.2f}), \
        Unseen Acc = ({accs_unseen[idx_best_H]*100:.2f} +/- {accs_unseen_std[idx_best_H]*100:.2f})%, Seen Acc = ({accs_seen[idx_best_H]*100:.2f} +/- {accs_seen_std[idx_best_H]*100:.2f})%, averaged over seeds {opt.manualSeed}")
        
    print("All experiments over the list of seeds completed.")
    print("-------------------------")
