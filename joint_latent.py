import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import utility.util as util 
import copy
import os
from torch.utils.data import Dataset, DataLoader
import torchvision

from regressor import REGRESSOR
import utility.model_bases as model

from baselines.wDAEGNN.low_shot_learning.architectures.classifiers.weights_denoising_autoencoder import WeightsDAE
import utility.model_bases as model


class Joint(REGRESSOR):
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base=False, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, _embed_dim=1000, _num_layers=3, opt=None):
        super().__init__(_train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base, _lr, _beta1, _nepoch, _batch_size, _embed_dim, _num_layers, opt)
        self.opt = opt
        self.seedinfo = seedinfo
        
        self.lr = _lr
        self.beta1 = _beta1
        self.nepoch = _nepoch
        self.batch_size = _batch_size
        self.embed_dim = _embed_dim
        self.num_layers = _num_layers
        self.nclass = _nclass
        self.cuda = _cuda

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses            
        self.attribute = data_loader.attribute

        if self.opt.subspace_proj:
            base_weights_mat = torch.cat((self.model.fc.weight.data, self.model.fc.bias.data.unsqueeze(1)), 1)
            tr_base = torch.transpose(base_weights_mat, 0, 1)
            self.Q, self.R = torch.linalg.qr(tr_base, mode='reduced')
        
        if self.opt.cuda:
            self.target_weights = self.target_weights.cuda()
        
        if opt.class_reduction_ablation:
            perm = torch.randperm(len(self.seenclasses))
            assert opt.class_reduction_ablation in range(1, len(self.seenclasses)+1)
            perm = perm[:opt.class_reduction_ablation]
            training_attributes = self.attribute[self.seenclasses][perm]
            training_weights = self.target_weights[perm]

        if opt.single_autoencoder_baseline:
            if opt.class_reduction_ablation: 
                att2weight_dataset = util.GenericDataset(opt, _input=training_attributes, _target=training_weights, cuda=self.cuda)
                self.loader = DataLoader(att2weight_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                att2weight_dataset = util.GenericDataset(opt, _input=self.attribute[self.seenclasses], _target=self.target_weights, cuda=self.cuda)
                self.loader = DataLoader(att2weight_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            if opt.class_reduction_ablation:
                combined_seen_dataset = util.GenericDataset(opt, _input=training_attributes, _target=training_weights, cuda=self.cuda)
            else:
                combined_seen_dataset = util.GenericDataset(opt, _input=self.attribute[self.seenclasses], _target=self.target_weights, cuda=self.cuda)
            placeholder_weights = torch.zeros(len(self.unseenclasses), self.target_weights.size(1))
            if self.cuda:
                placeholder_weights = placeholder_weights.cuda()
            if opt.class_reduction_ablation:
                combined_full_dataset = util.GenericDataset(opt, _input=torch.cat((training_attributes, self.attribute[self.unseenclasses])), _target=torch.cat((training_weights, placeholder_weights), dim=0), cuda=self.cuda)
            else:
                combined_full_dataset = util.GenericDataset(opt, _input=torch.cat((self.attribute[self.seenclasses], self.attribute[self.unseenclasses])), _target=torch.cat((self.target_weights, placeholder_weights), dim=0), cuda=self.cuda)
            if self.opt.include_unseen:
                self.loader = DataLoader(combined_full_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                self.loader = DataLoader(combined_seen_dataset, batch_size=self.batch_size, shuffle=True)

        self.AE_attribute = model.AUTOENCODER(self.opt, input_dim=self.attribute.size(1), embed_dim=self.embed_dim, num_layers=self.num_layers)
        self.AE_weight = model.AUTOENCODER(self.opt, input_dim=self.target_weights.size(1), embed_dim=self.embed_dim, num_layers=self.num_layers)
        
        self.AE_attribute.apply(util.weights_init)
        self.AE_weight.apply(util.weights_init)
        
        if self.opt.single_autoencoder_baseline:
            self.model = model.AUTOENCODER(self.opt, input_dim=self.attribute.size(1), embed_dim=self.embed_dim, output_dim=self.target_weights.size(1), num_layers=self.num_layers)
        else:
            self.model = model.JOINT_AUTOENCODER(self.opt, autoencoder1=self.AE_attribute, autoencoder2=self.AE_weight)

        if self.model:
            self.model.apply(util.weights_init)
        
        if opt.cos_sim_loss:
            self.criterion = util.cos_sim_loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1loss = nn.L1Loss(reduction='none')

        if opt.daegnn:
            dae_num_features = 2049 # Number of features from ResNet-101, 512 in original implementation.
            self.dae_meta_batch_size = 4 # Taken from original implementation
            if self.opt.single_autoencoder_baseline:
                self.dae_loader = DataLoader(att2weight_dataset, batch_size=len(self.seenclasses), shuffle=False)
            else:
                self.dae_loader = DataLoader(combined_seen_dataset, batch_size=len(self.seenclasses), shuffle=False)
            self.dae = WeightsDAE({
                'gaussian_noise': 0.08,
                'comp_reconstruction_loss': True,
                'targets_as_input': False,
                'dae_type': 'RelationNetBasedGNN',
                'num_layers': 2,
                'num_features_input': dae_num_features,
                'num_features_output': 2 * dae_num_features,
                'num_features_hidden': 3 * dae_num_features,
                'update_dropout': 0.7,

                'nun_features_msg': 3 * dae_num_features,
                'aggregation_dropout': 0.7,
                'topK_neighbors': 10,
                'temperature': 5.0,
                'learn_temperature': False,
            })
            self.dae_optimizer = optim.Adam(self.dae.parameters(), lr=_lr, betas=(_beta1, 0.999), weight_decay=0.0)
        
        if self.cuda:
            self.AE_attribute.cuda()
            self.AE_weight.cuda()
            if self.model:
                self.model.cuda()
            self.criterion.cuda()
            self.mse_loss.cuda()
            self.l1loss.cuda()
            if opt.daegnn:
                self.dae.cuda()

        self.unseen_model = model.LINEAR(self.test_seen_feature.size(1), len(self.unseenclasses))
        self.ext_model = model.LINEAR(self.test_seen_feature.size(1), len(self.seenclasses) + len(self.unseenclasses))
        self.ext_model.fc.weight.data[:len(self.seenclasses), :] =  self.target_weights[:, :-1]
        self.ext_model.fc.bias.data[:len(self.seenclasses)] = self.target_weights[:, -1]
                
        if self.cuda:
            self.ext_model.cuda()
            self.unseen_model.cuda()

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer_attribute_AE = optim.Adam(self.AE_attribute.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.optimizer_weight_AE = optim.Adam(self.AE_weight.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.model:
            self.weight_optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999), weight_decay=0.0)
    
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
    def fit(self):
        run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = 0, 0, 0, 0, 0
        relu = torch.nn.ReLU()
        
        self.att_std = torch.std_mean(self.attribute, dim=0)[0].cpu()
        self.weight_std = torch.std_mean(self.target_weights, dim=0)[0].cpu()

        counter = 0
        breaking = False
        epoch_losses = []
        
        for epoch in range(self.nepoch):
            epoch_loss = 0
            epoch_att_from_att_loss = 0
            epoch_att_from_weight_loss = 0
            epoch_weight_from_weight_loss = 0
            epoch_weight_from_att_loss = 0
            epoch_alignment_loss = 0
            
            for i_batch, batch in enumerate(self.loader):
                # Create mask to remove loss from weight prediction from unseen class attributes
                mask = torch.where(torch.sum(torch.abs(batch[1]), dim=-1) > 0., 1., 0.)[:, None]
                if self.cuda:
                    mask = mask.cuda()
                mask_sum = torch.clamp(mask.sum(), min=1.)
                inv_mask_sum = torch.clamp((1-mask).sum(), min=1)
                    
                self.model.zero_grad()
                
                if self.opt.single_autoencoder_baseline:
                    att, weights = batch
                    output = self.model(att)
                    loss = self.criterion(output, weights)
                    loss = loss.mean()

                    if self.opt.subspace_proj: 
                        mut = output @ self.Q
                        mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                        proj_weights = mutnorm @ self.Q.T
                        proj_weights = proj_weights.squeeze()
                        subspace_proj_loss = 0.001 * torch.norm(output - proj_weights, dim=-1).mean()
                        loss += subspace_proj_loss

                else:
                    output = self.model(batch)
                    att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = output

                    att_from_att_loss = self.criterion(att_from_att, batch[0]).mean() 
                    att_from_weight_loss = (self.criterion(att_from_weight, batch[0])*mask).sum(0).mean()/mask_sum
                    if self.opt.single_modal_ablation:
                        att_from_weight_loss = 0 * att_from_weight_loss
                    weight_from_weight_loss = (self.criterion(weight_from_weight, batch[1])*mask).sum(0).mean()/mask_sum 
                    weight_from_att_loss = (self.criterion(weight_from_att, batch[1])*mask).sum(0).mean()/mask_sum 
                    
                    loss = att_from_att_loss + att_from_weight_loss + weight_from_weight_loss + weight_from_att_loss

                    epoch_att_from_att_loss += att_from_att_loss.data     
                    epoch_att_from_weight_loss += att_from_weight_loss.data     
                    epoch_weight_from_weight_loss += weight_from_weight_loss.data     
                    epoch_weight_from_att_loss += weight_from_att_loss.data       

                    if self.opt.subspace_proj:
                        mut = weight_from_att @ self.Q
                        mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                        proj_weights = mutnorm @ self.Q.T
                        proj_weights = proj_weights.squeeze()
                        subspace_proj_loss = 0.001 * torch.norm(weight_from_att - proj_weights)
                        loss += subspace_proj_loss

                epoch_loss += loss.data                       

                loss.backward()
                self.weight_optimizer.step()
                
            epoch_loss /= len(self.loader)
            epoch_losses.append(epoch_loss)
            if epoch == 0:
                prev_loss = epoch_loss
            else:
                loss_diff = torch.abs(prev_loss - epoch_loss)
                prev_loss = epoch_loss
            
            if self.opt.single_autoencoder_baseline:
                epoch_info = {"loss": epoch_loss}
            else:
                epoch_info = {"loss": epoch_loss, "att_from_att_loss": epoch_att_from_att_loss, 
                                "att_from_weight_loss": epoch_att_from_weight_loss, "weight_from_weight_loss": epoch_weight_from_weight_loss, 
                                "weight_from_att_loss": epoch_weight_from_att_loss}
            
            if self.opt.early_stopping_slope:
                if epoch > 20:
                    threshold = 2 * 10e-4 if self.opt.cos_sim_loss else 2 * 10e-7 
                    slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(torch.stack(epoch_losses)[-20:-10])) / 10.
                    if slope < threshold:
                        counter += 1
                        if counter == 5:
                            breaking = True
                    else:
                        counter = 0
                    epoch_info["slope"] = slope 

            # Check down-stream performance (ZSL or GZSL) of weights predicted by current network state.
            # Note that performances seen here cannot be reported, as we are implicitly assuming access to images during training to do this.
            if ((not self.opt.strict_eval) or (epoch + 1 == self.nepoch) or breaking) and not self.opt.daegnn:
                self.model.eval()
                if epoch + 1 == self.nepoch or breaking:
                    self.calc_entropy = self.opt.calc_entropy
                    
                val_out = self.pred_weights_and_val(weight_model=self.model)
                self.model.train()

                if self.opt.zst:
                    acc_target, acc_zst_unseen = val_out
                else:
                    acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                    epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                    epoch_info["H"] = H
                    epoch_info["acc_unseen_gzsl"] = acc_unseen
                    epoch_info["acc_seen_gzsl"] = acc_seen
                        
                    # Save best performing downstream model
                    if H >= run_best_H:
                        run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl
                        best_weight_model = copy.deepcopy(self.model)
            
            if breaking:
                print("Stopping early (slope criterion)")
                break
            
        
        if self.opt.daegnn:
            print("Starting training of wDAE-GNN")
            comp_loss = nn.MSELoss(reduction='none')
            counter = 0
            breaking = False
            epoch_losses = []
            for epoch in range(self.nepoch):
                epoch_loss = 0
                for _, batch in enumerate(self.dae_loader):
                    self.model.zero_grad()
                    self.dae.zero_grad()
                    
                    if self.opt.single_autoencoder_baseline:
                        att, weights = batch
                        output = self.model(att).detach()
                        perm = torch.randperm(weights.size(0))
                        weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

                        num_idxs = weights.size(0) // self.dae_meta_batch_size
                        for i in range(self.dae_meta_batch_size):
                            idx = perm[i*num_idxs:(i+1)*num_idxs]
                            weights_input[i][idx] = output[idx]

                        recon = self.dae(weights_input)
                        loss = comp_loss(recon, weights_input).mean()
                        loss.backward()
                        self.dae_optimizer.step()
                    else:
                        att, weights = batch
                        output = self.model(batch)
                        att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = output
                        perm = torch.randperm(weights.size(0))
                        weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

                        num_idxs = weights.size(0) // self.dae_meta_batch_size
                        for i in range(self.dae_meta_batch_size):
                            idx = perm[i*num_idxs:(i+1)*num_idxs]
                            weights_input[i][idx] = weight_from_att[idx]

                        recon = self.dae(weights_input)
                        loss = comp_loss(recon, weights_input).mean()
                        loss.backward()
                        self.dae_optimizer.step()

                    epoch_loss += loss.data

                epoch_loss /= len(self.dae_loader)
                epoch_losses.append(epoch_loss)
                if epoch == 0:
                    prev_loss = epoch_loss
                else:
                    loss_diff = torch.abs(prev_loss - epoch_loss) 
                    prev_loss = epoch_loss
                
                if self.opt.single_autoencoder_baseline:
                    epoch_info = {"loss": epoch_loss}

                else:
                    epoch_info = {"loss": epoch_loss, "att_from_att_loss": epoch_att_from_att_loss, 
                                    "att_from_weight_loss": epoch_att_from_weight_loss, "weight_from_weight_loss": epoch_weight_from_weight_loss, 
                                    "weight_from_att_loss": epoch_weight_from_att_loss}

                if self.opt.early_stopping_slope:
                    if epoch > 20:
                        threshold = 2 * 10e-4 if self.opt.cos_sim_loss else 2 * 10e-7 
                        slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(torch.stack(epoch_losses)[-20:-10])) / 10.
                        if slope < threshold:
                            counter += 1
                            if counter == 5:
                                breaking = True
                        else:
                            counter = 0
                        epoch_info["slope"] = slope 

                # Check down-stream performance (ZSL or GZSL) of weights predicted by current network state.
                # Note that performances seen here cannot be reported, as we are implicitly assuming access to images during training to do this.
                if (not self.opt.strict_eval) or (epoch + 1 == self.nepoch) or breaking:
                    self.model.eval()
                    self.dae.eval()
                    if epoch + 1 == self.nepoch or breaking:
                        self.calc_entropy = self.opt.calc_entropy
                    
                    val_out = self.pred_weights_and_val(weight_model=self.model, daegnn=self.dae)
                    self.model.train()
                    self.dae.train()

                    if self.opt.zst:
                        acc_target, acc_zst_unseen = val_out
                    else:
                        acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                        epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                        epoch_info["H"] = H
                        epoch_info["acc_unseen_gzsl"] = acc_unseen
                        epoch_info["acc_seen_gzsl"] = acc_seen
                            
                        # Save best performing downstream model
                        if H >= run_best_H:
                            print("New best GZSL based on H (seed):", H)
                            run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl
                            best_weight_model = copy.deepcopy(self.model)

                if breaking:
                    print("Stopping early")
                    break
        
        if self.opt.zst:
            self.acc_target, self.acc_zst_unseen = acc_target, acc_zst_unseen
        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl
