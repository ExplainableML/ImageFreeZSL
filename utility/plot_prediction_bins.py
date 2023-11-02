import argparse
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

plt.rcParams.update({ "text.usetex":True, "font.family": "serif"})

def create_dict_foldernum_to_label(matcontent_splits):
    cls_names = matcontent_splits['allclasses_names']
    folder_to_label_dict = {} 
    for n, name in enumerate(cls_names):
        folder_to_label_dict[name[0][0]] = n
    return folder_to_label_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='FLO')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode (Deprecated)')
parser.add_argument('--dataroot', default='/home/shared/iccv-apzsl/Feature-Generation-datasets', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='tf_finetune')
parser.add_argument('--class_embedding', default='att')

opt = parser.parse_args()
opt.zst = False
assert opt.baseline in ['SMO', 'ConSE']

matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

feature = matcontent['features'].T
if 'CUB_' in opt.image_embedding:
        feature = feature.T
label = matcontent['labels'].astype(int).squeeze() - 1
mat_path = opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat"
matcontent = sio.loadmat(mat_path)
print("using the matcontent:", mat_path)

trainval_loc = matcontent['trainval_loc'].squeeze() - 1
train_loc = matcontent['train_loc'].squeeze() - 1
val_unseen_loc = matcontent['val_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

attribute = torch.from_numpy(matcontent['att'].T).float() 

folder_to_label_dict = create_dict_foldernum_to_label(matcontent)
label_to_folder_dict = {v: k for k, v in folder_to_label_dict.items()}

train_feature = torch.from_numpy(feature[trainval_loc]).float()
train_label = torch.from_numpy(label[trainval_loc]).long() 
test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

seenclasses = torch.from_numpy(np.unique(train_label.numpy()))
unseenclasses = torch.from_numpy(np.unique(test_unseen_label.numpy()))
ntrain = train_feature.size()[0]
ntrain_class = seenclasses.size(0)
ntest_class = unseenclasses.size(0)
train_class = seenclasses.clone()
allclasses = torch.arange(0, ntrain_class+ntest_class).long()

per_cls_acc_gzsl = torch.load('nounseen_percls_acc_CUBtf_finetune_len_test_4731_len_tar_200.pt')
per_cls_acc_unseen_zsl = torch.load('nounseen_percls_acc_CUBtf_finetune_len_test_2967_len_tar_50.pt')
pred_matrix_gzsl = torch.load('nounseen_pred_matrixCUBtf_finetune_len_test_4731_len_tar_200.pt')
pred_matrix_unseen_zsl = torch.load('nounseen_pred_matrixCUBtf_finetune_len_test_2967_len_tar_50.pt')

# GZSL plot 
min_acc, min_idx = torch.min(per_cls_acc_gzsl[-len(unseenclasses):], dim=0)
min_idx = min_idx + len(seenclasses)
low_row = pred_matrix_gzsl[min_idx, :]
low_confuse = low_row.nonzero().squeeze()
low_confuse = low_confuse[low_confuse != min_idx]
labels = torch.cat((seenclasses, unseenclasses))
low_confuse_labels = []
for idx in low_confuse:
        low_confuse_labels.append(labels[idx].numpy())
low_confuse_labels = np.array(low_confuse_labels)

low_acc_cls = str(label_to_folder_dict[int(labels[min_idx].numpy())])[4:].replace("_", " ")
print("Ours, GZSL, Low acc class:", low_acc_cls, "with probability:", min_acc)
for label in low_confuse_labels:
        print(label_to_folder_dict[int(label)][4:].replace("_", " "))

# Ordering based on attribute cosine similarity
min_att = attribute[labels[min_idx]]
cosine_sims = []
cos = nn.CosineSimilarity(dim=0, eps=1e-8)
attributes = torch.cat((attribute[seenclasses], attribute[unseenclasses]))
for att in attributes:
        cosine_sims.append(cos(min_att, att))
cosine_sims = torch.stack(cosine_sims)
cosine_sims_ordered, cosine_sim_indeces = torch.topk(cosine_sims, k=len(attributes))

ordered_probs = low_row[cosine_sim_indeces]
ordered_labels = labels[cosine_sim_indeces]
ordered_cls_names = []
bar_colors = []
line_styles = []
for label in ordered_labels:
        cls_name = label_to_folder_dict[int(label)][4:].replace("_", " ")
        if label in unseenclasses:
                bar_colors.append('mediumblue')
                line_styles.append('--')
        else:
                bar_colors.append('chocolate')
                line_styles.append('-')
        ordered_cls_names.append(cls_name)

ordered_cls_names = np.array(ordered_cls_names)

# Bar plot of bins of similar classes (course)
fig = plt.figure(figsize = (10, 5)) 
bin_size = 10
bin_labels = [f'Rank {int(x-bin_size) + 1} to {int(x)}' for x in bin_size * np.arange(start=1, stop=len(ordered_probs)/bin_size+1)]
binned_probs = np.squeeze(np.sum(np.reshape(ordered_probs.numpy(), (len(ordered_probs)//bin_size, bin_size)), axis=1))
barlist = plt.bar(bin_labels, binned_probs, color ='mediumblue',
        width = 0.4)
 
plt.xlabel("CUB classes ordered by similarity with " + low_acc_cls, fontsize = 16)
plt.xticks(rotation=45, ha='right')

plt.ylabel("Fraction of predictions", fontsize = 16)
plt.savefig('overview_low_acc_class.png',
            format='png',
            dpi=1600,
            bbox_inches='tight')


# Bar plot of n most similar classes (finegrained)
fig = plt.figure(figsize = (10, 5))
nonzero_idxs = np.array(ordered_probs.nonzero().squeeze())
barlist = plt.bar(ordered_cls_names[nonzero_idxs], ordered_probs.numpy()[nonzero_idxs], color ='mediumblue', width = 0.4)

bar_colors = np.array(bar_colors)[nonzero_idxs]
line_styles = np.array(line_styles)[nonzero_idxs]
for n, ls in enumerate(line_styles):
        if ls == '--':
                barlist[n].set_color('w')
        barlist[n].set_linewidth(4)
        barlist[n].set_linestyle(ls)
        barlist[n].set_edgecolor('mediumblue')
        
plt.xlabel("Classes ordered by similarity to " + low_acc_cls, fontsize = 16)
plt.xticks(rotation=45, ha='right')

colors = {'Seen class':'mediumblue', 'Unseen class':'w'}     
linestyles = {'Seen class':'-', 'Unseen class':'--'}    
edgecolors = {'Seen class':'mediumblue', 'Unseen class':'mediumblue'}      
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[label], linewidth=1, linestyle=linestyles[label], edgecolor=edgecolors[label]) for label in labels]
plt.legend(handles, labels)

plt.ylabel("Fraction of predictions", fontsize = 16)
plt.savefig('low_acc_class.png',
            format='png',
            dpi=1600,
            bbox_inches='tight')
