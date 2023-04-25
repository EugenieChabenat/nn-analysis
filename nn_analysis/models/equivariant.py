import torch
import torch.nn as nn
import torchvision.models as models

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        #sizes = [2048] + list(map(int, args.projector.split('-')))
        sizes = [32] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.projector_sizes = sizes

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        loss = on_diag + self.args.lambd * off_diag
        return loss
# ----

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)

      x += identity
      x = self.relu(x)
      return x

# -- -- -- -- 
import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision.models
# --
import argparse
from models import BarlowTwins

def get_parser():
    parser = argparse.ArgumentParser(description='Model-specific parameters')
    parser.add_argument('--pos-weight', default=0.03, type=float,
                        help='cam pos targets will be multiplied by this number')
    parser.add_argument('--scale-weight', default=20.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    parser.add_argument('--color-weight', default=15.0, type=float,
                        help='scale pos targets will be multiplied by this number')
    return parser
        
        

    
### Utilities ###

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
# -- 


# -- -- -- to be moved to an other file for clarity / readability
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)

      x += identity
      x = self.relu(x)
      return x

# -- -- -- -- 
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3): # is adding args going to work? 
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        #self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        #self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # change the shape of the linear layer to adapt to the fact there is not the rest of the network anymore 
        #self.fc = nn.Linear(512*ResBlock.expansion, num_classes) # original line of code 
        self.fc = nn.Linear(512, 32)
        
    def forward(self, x, delta_vec):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        
        #print('after: ', x.shape)
   
        # delta is added after V1 
        #x = x + delta_vec 
        
        # Note: for V1 evaluation, do not need to go through the rest of the network 
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        
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

        
def custom_resnet50(num_classes, delta_vec, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, delta_vec, channels)

# -- Model class
class Model(BarlowTwins):
    def __init__(self, args):
        super().__init__(args)
        
        # original size 
        sizes = {'bn': torch.Size([self.projector_sizes[-1]])}
                
        self.custom_resnet = ResNet(Bottleneck, [3,4,6,3], num_classes=2048, num_channels=3)
        
        #sizes_ = {'bn': torch.Size([self.custom_resnet.layer1])}
        
        # create embedding matrix
        self.register_buffer("embedding", nn.init.orthogonal_(torch.empty(7, *sizes['bn'])))
        #self.register_buffer("embedding", nn.init.orthogonal_(torch.empty(7, 56)))
        
        self.pos_weight = args.pos_weight
        self.scale_weight = args.scale_weight
        self.color_weight = args.color_weight
        
        
        #self.custom_resnet = ResNet(Bottleneck, [3,4,6,3], num_classes=1000, num_channels=3)
        
    def forward(self, y1, y2, **kwargs):
        # args and kwargs should be on gpu
        delta_pos = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_pos_x', 'cam_pos_y']], dim=1)*self.pos_weight
        delta_scale = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['cam_scale']], dim=1)*self.scale_weight
        delta_color = torch.cat([kwargs[f'{key}_1'] - kwargs[f'{key}_0'] for key in ['brightness', 'contrast', 'saturation', 'hue']], dim=1)*self.color_weight
        is_not_bw = ((1.0-kwargs['applied_RandomGrayscale_0'])*(1.0-kwargs['applied_RandomGrayscale_1'])).squeeze()
        is_color_jittered = (kwargs['applied_ColorJitter_0']*kwargs['applied_ColorJitter_1']).squeeze()
        delta_color = torch.einsum('b,bm->bm',is_not_bw*is_color_jittered,delta_color)
        
        delta = torch.cat([delta_pos, delta_scale, delta_color], dim=1)
        
        # multiplication with a semi orthogonal matrix A
        delta_vec = torch.cat([delta_pos, delta_scale, delta_color], dim=1).matmul(self.embedding)
        
        
                
        # -- replace computations of z1 with the custom resnet class   
        z1 = self.bn(self.projector(self.custom_resnet(y1, delta_vec)))
        #print('z1: ', z1.shape)
        # -- 
        
        # -- this has to be removed since the delta is injected before
        #z1 = self.bn(self.projector(self.backbone(y1)))
        z1 = z1 + delta_vec
        # --
        # z2 should also stop after the first layer 
        #z2 = self.bn(self.projector(self.backbone(y2)))
        z2 = self.bn(self.projector(self.custom_resnet(y2, delta_vec)))
        
        # -- the rest is not supposed to change --
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
        loss = on_diag + self.args.lambd * off_diag
        
        # -- added
        loss_components = {'contrastive': loss}
        # -- 
        
        return loss, loss_components
    
    def get_encoder(self):
        return self.backbone
