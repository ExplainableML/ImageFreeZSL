import torch.nn as nn
import torch

class LINEAR(nn.Module):
    def __init__(self, input_dim, nclass, bias=True):
        super(LINEAR, self).__init__()
        self.fc = nn.Linear(input_dim, nclass, bias)
    def forward(self, x): 
        o = self.fc(x)
        return o

class LINEAR_TO_COS_SIM(nn.Module):
    def __init__(self, weights):
        super(LINEAR, self).__init__()
        self.weights = weights
        self.cos = nn.functional.cosine_similarity(dim=1)
    def forward(self, x): 
        out = []
        for sample in x:
            temp = []
            for weight in self.weights:
                temp.append(self.cos(weight, sample))
            out.append(torch.stack(temp))
        o = torch.stack(out)
        return o  

class WEIGHT_PREDICTOR(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_layers=3):
        super(WEIGHT_PREDICTOR, self).__init__()
        assert num_layers in [1, 2, 3]
        self.num_layers = num_layers
        if num_layers == 4:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.fc3 = nn.Linear(embed_dim, embed_dim)
            self.fc4 = nn.Linear(embed_dim, output_dim)
        if num_layers == 3:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.fc3 = nn.Linear(embed_dim, output_dim)
        elif num_layers == 2:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x): 
        if self.num_layers == 4:
            o = self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))
        elif self.num_layers == 3:
            o = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        elif self.num_layers == 2:
            o = self.fc2(self.relu(self.fc1(x)))     
        else:
            o = self.fc1(x)
        return o  
        

class AUTOENCODER(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, output_dim=None, num_layers=3, vae=False, bias=True):
        super(AUTOENCODER, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is None:
            self.output_dim = input_dim   
        if vae: 
            self.embed_dim = [2 * embed_dim, embed_dim]
        else:
            self.embed_dim = [embed_dim, embed_dim]
        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
                )
                
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], self.output_dim)
                )
        if num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
                )
                
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
                )
        if num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
                )
                
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
                )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)


class ATT_AUTOENCODER(AUTOENCODER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1450),
            nn.ReLU(inplace=True),
            nn.Linear(1450, self.embed_dim),
            nn.ReLU(inplace=True)
            )
            
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 660),
            nn.ReLU(inplace=True),
            nn.Linear(660, self.output_dim)
            )

class WEIGHT_AUTOENCODER(AUTOENCODER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1560),
            nn.ReLU(inplace=True),
            nn.Linear(1560, self.embed_dim),
            nn.ReLU(inplace=True)
            )
            
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 1660),
            nn.ReLU(inplace=True),
            nn.Linear(1660, self.output_dim)
            )


class JOINT_AUTOENCODER(nn.Module):
    def __init__(self, opt, autoencoder1, autoencoder2):
        super(JOINT_AUTOENCODER, self).__init__()
        self.ae1 = autoencoder1
        self.ae2 = autoencoder2
            
    def encode1(self, x): 
        return self.ae1.encode(x)

    def encode2(self, x):
        return self.ae2.encode(x)

    def decode1(self, x):
        return self.ae1.decode(x)

    def decode2(self, x):
        return self.ae2.decode(x)
    
    def forward(self, x): 
        att_in, weight_in = x
        latent_att = self.encode1(att_in)
        latent_weight = self.encode2(weight_in)
        
        att_from_att = self.decode1(latent_att)
        att_from_weight = self.decode1(latent_weight)
        weight_from_weight = self.decode2(latent_weight)
        weight_from_att = self.decode2(latent_att)
        
        return att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight

    def predict(self, x):
        # Given attributes, predict weights
        latent_att = self.encode1(x)
        return self.decode2(latent_att)


class VAE(AUTOENCODER):
    def __init__(self, *args, **kwargs):
        super().__init__(vae=True, *args, **kwargs)
    
    def reparameterize(self, mu, logvar, noise=True):
        if noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            if self.opt.cuda:
                eps = eps.cuda()
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def encode(self, x):
        out = self.encoder(x)
        out = torch.split(out,out.shape[1]//2, dim=1)
        mu_batch, logvar_batch = out[0], out[1]
        kl_div = (0.5 * torch.sum(1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp()))
        return self.reparameterize(mu_batch, logvar_batch), kl_div, mu_batch, logvar_batch
        
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z, kl_div, mu, logvar, = self.encode(x)
        return self.decoder(z), kl_div, mu, logvar

class JOINT_VAE(JOINT_AUTOENCODER):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
    
    def forward(self, x): 
        att_in, weight_in = x
        (z_att, kl_div_att, mu_att, logvar_att) = self.encode1(att_in)
        (z_weight, kl_div_weight, mu_weight, logvar_weight) = self.encode2(weight_in)
        att_from_att = self.decode1(z_att)
        att_from_weight = self.decode1(z_weight)
        weight_from_weight = self.decode2(z_weight)
        weight_from_att = self.decode2(z_att)

        return att_from_att, att_from_weight, weight_from_weight, weight_from_att, z_att, z_weight, kl_div_att, kl_div_weight, mu_att, logvar_att, mu_weight, logvar_weight

    def predict(self, x):
        # Given attributes, predict weights
        (z_att, kl_div_att, mu_att, logvar_att) = self.encode1(x)
        return self.decode2(z_att)
    