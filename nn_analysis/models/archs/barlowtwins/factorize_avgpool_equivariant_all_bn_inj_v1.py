import torch
import torch.nn as nn
import argparse

from models import BarlowTwins
from models import Bottleneck, Block 
from argparse import Namespace

import torch
import torch.nn as nn

from .base import BarlowTwins

def get_parser():
    parser = argparse.ArgumentParser(description='Model-specific parameters')
    parser.add_argument('--queue-size', default=8192, type=int,
                        help='number of images stored in queue for computing factorization score')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='explained variance ratio threshold for image subspace')
    parser.add_argument('--pos-weight', default=0.03, type=float,
                        help='cam pos targets will be multiplied by this number')
    parser.add_argument('--scale-weight', default=20.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    parser.add_argument('--color-weight', default=15.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    parser.add_argument('--equivariant-weight', default=1.0, type=float,
                        help='equivariant loss will be multiplied by this number')
    parser.add_argument('--factorization-weight', default=0.2, type=float,
                        help='factorization loss will be multiplied by this number')
                        
    # -- added 
    parser.add_argument('--injection_site', default='V1', type=str,
                        help='where to inject and evaluate')
                        
    return parser

# -- added 

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, injection_site='V1', num_channels=3): # is adding args going to work? 
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        # -- added 
        self.injection_site = injection_site
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        proj_dim = 256
        if self.injection_site!='V1': 
            self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
            proj_dim = 512
            if self.injection_site !='V2':
                self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
                proj_dim = 1024
                if self.injection_site=='IT':
                    self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
                    proj_dim = 512*ResBlock.expansion
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # change the shape of the linear layer to adapt to the fact there is not the rest of the network anymore 
        #self.fc = nn.Linear(512*ResBlock.expansion, num_classes) # original line of code
        
        self.fc = nn.Linear(proj_dim, 32)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        
        # Note: for V1 evaluation, do not need to go through the rest of the network 
        
        if self.injection_site!='V1': 
            x = self.layer2(x)
            if self.injection_site !='V2':
                x = self.layer3(x)
                if self.injection_site=='IT':
                    x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

      
# -- 


class Model(BarlowTwins):
    def __init__(self, args):
        super().__init__(args)
        
        #sizes = {'bn': torch.Size([self.projector_sizes[-1]]), 'avgpool': torch.Size((2048,))}
        sizes = {'bn': torch.Size([self.projector_sizes[-1]]), 'avgpool': torch.Size((32,))}
        
        # -- added 
        self.injection_site = args.injection_site 
        self.backbone = ResNet(Bottleneck, [3,4,6,3], num_classes=2048, injection_site=self.injection_site, num_channels=3)
        # -- 
        
        # create embedding matrix
        self.register_buffer("embedding", nn.init.orthogonal_(torch.empty(7, *sizes['bn'])))
        
        # factorization
        self.queue_size = args.queue_size
        self.register_buffer("queue", torch.randn(*sizes['avgpool'], self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.pca = PCA(threshold=args.threshold)
        
        # criterions
        self.criterions = {
            'factorize': MSELoss(target_weights=[2.0]).cuda(self.args.gpu),
            'pos': MSELoss(target_weights=[2.0,2.0]).cuda(self.args.gpu),
            'scale': MSELoss().cuda(self.args.gpu),
            'color': MSELoss().cuda(self.args.gpu),
        }
        
    def forward(self, y1, y2, **kwargs):
        # args and kwargs should be on gpu
        delta_pos = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_pos_x', 'cam_pos_y']], dim=1)*self.args.pos_weight
        delta_scale = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_scale']], dim=1)*self.args.scale_weight
        delta_color = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['brightness', 'contrast', 'saturation', 'hue']], dim=1)*self.args.color_weight
        is_not_bw = ((1.0-kwargs['applied_RandomGrayscale_0'])*(1.0-kwargs['applied_RandomGrayscale_1'])).squeeze()
        is_color_jittered = (kwargs['applied_ColorJitter_0']*kwargs['applied_ColorJitter_1']).squeeze()
        delta_color = torch.einsum('b,bm->bm',is_not_bw*is_color_jittered,delta_color)
        
        delta_vec = torch.cat([delta_pos, delta_scale, delta_color], dim=1).matmul(self.embedding)
        x1 = self.backbone(y1)
        z1 = self.bn(self.projector(x1))
        z1 = z1 + delta_vec
        x2 = self.backbone(y2)
        z2 = self.bn(self.projector(x2))
        
        # -- added (should not be necessary since backbone is redefined)
        #x1 = 
        # -- 
        
        # factorization
        with torch.no_grad():
            self.pca.fit(self.queue.t()) # fact_queue.t() - (queue_size, n_features)
        acts = torch.stack([x1,x2], dim=0)
        acts_centered = (acts - acts.mean(dim=0)).reshape(-1,acts.size()[-1]) # (2*batch_size,n_features)
        
        #print('acts_centered: ', acts_centered.shape)
        
        acts_centered_proj = self.pca.transform(acts_centered)
        self._dequeue_and_enqueue(acts.mean(dim=0))
        
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        equivariant_loss = on_diag + self.args.lambd * off_diag
        
        # get output and target; args and kwargs should be on gpu
        output = {
            'factorize': acts_centered_proj.var(dim=0).sum()/acts_centered.var(dim=0).sum(),
        }
        
        target = {
            'factorize': torch.zeros(1).cuda(self.args.gpu),
        } 
        
        # get total loss
        losses = {
            'Equivariant Loss': equivariant_loss,
            'Factorization Loss': self.criterions['factorize'](output['factorize'], target['factorize']),
        }
        loss_components = {name: loss.item() for name, loss in losses.items()}
        weights = torch.Tensor([
            self.args.equivariant_weight,
            self.args.factorization_weight,
        ]).cuda(self.args.gpu)
        
        loss = weights.dot(torch.stack([loss.squeeze() for loss in losses.values()]))
        return loss, loss_components
    
    def get_encoder(self):
        return self.backbone
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.queue_size % batch_size == 0:  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr
        else:
            print("Incompatible batch, skipping queue update")
    
### Utilities ###

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class PCA():
    def __init__(self, threshold=0.9):
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold
        
    def fit(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/decomposition/_pca.py
        """
        assert X.ndim == 2
        n_samples, n_features = X.size()[0], X.size()[1]
        
        self.mean_ = X.mean(dim=0)
        
        print('X: ', X.shape)
        print('mean: ', self.mean_.shape)
        
        self.mean_ = X.mean(dim=0)
        X = X - self.mean_
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        components_ = Vt
        
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        
        ratio_cumsum = torch.cumsum(explained_variance_ratio_, dim=0)
        n_components = torch.searchsorted(ratio_cumsum, self.threshold, right=True) + 1
        
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        
    def transform(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/decomposition/_base.py
        """
        X = X - self.mean_
        return X.mm(self.components_.t())
    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MSELoss(nn.Module):
    def __init__(self, target_weights=None):
        super().__init__()
        if target_weights is not None:
            target_weights = nn.Parameter(torch.Tensor(target_weights), requires_grad=False)
        self.register_parameter('target_weights', target_weights)
    
    def forward(self, output, target, sample_weights=None):
        if output.ndim == 0:
            output = output.unsqueeze(0).unsqueeze(1)
        if target.ndim == 0:
            target = target.unsqueeze(0).unsqueeze(1)
        if output.ndim == 1:
            output = output.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)
            
        batch_size, target_size = output.size(0), output.size(1)
        if self.target_weights is None:
            out = 0.5*((target-output)**2).mean(dim=1)
        else:
            out = 0.5*((target-output)**2).matmul(self.target_weights)/target_size
        if sample_weights is None:
            return out.mean()
        else:
            return out.matmul(sample_weights)/batch_size
        
def factorize_avgpool_equivariant_all_bn_inj_v1(projector='8192-8192-8192', batch_size=1024, scale_loss=0.024, lambd=0.0051, pos_weight=0.3, scale_weight=200.0, color_weight=150.0, **kwargs):
  args = Namespace(projector=projector, batch_size=batch_size, scale_loss=scale_loss, lambd=lambd, pos_weight=pos_weight, scale_weight=scale_weight, color_weight=color_weight, injection_site="V1", equivariant_weight = 1.0, factorization_weight= 0.2,  **kwargs)
  return Model(args)
