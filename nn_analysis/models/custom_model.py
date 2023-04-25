import torch

from nn_analysis.models import archs
from nn_analysis import utils
from nn_analysis import exceptions
from nn_analysis.constants import ENV_CONFIG_PATH
import disentangle.barlowtwins.models

import importlib
import argparse
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--name', type=str, metavar='NAME',
                    help='name of experiment')
parser.add_argument('--version', default=1, type=int, metavar='N',
                    help='version of model')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset on which to train the model')
parser.add_argument('--port-id', default=58472, type=int, metavar='N',
                    help='distributed training port number (default: 58472)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=0.024, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--tensorboard', action='store_true',
                    help='log training statistics to tensorboard')
parser.add_argument('--no-flip', action='store_true',
                    help='no random horizontal flips')

env_config = utils.load_config(ENV_CONFIG_PATH)

# -- 
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
        sizes = [2048] + list(map(int, args.projector.split('-')))
        #sizes = [32] + list(map(int, args.projector.split('-')))
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
# -- 

def _get_custom_model(arch, path=None, extract_method=None, model_kwargs={}, device='cpu', state_dict_key='state_dict'):
    archs_dict = {k: v for k, v in archs.__dict__.items() if not k.startswith("__") and callable(v) and k.islower()}
    model = archs_dict[arch](**model_kwargs)
    
    # --
    args, other_argv = parser.parse_known_args()
    args.name = "equivariant_all_bn_v2"
    other_args = importlib.import_module('.' + args.name, 'models').get_parser().parse_args(other_argv)
    args = argparse.Namespace(**vars(args), **vars(other_args))
    
    Model = importlib.import_module('.' + args.name, 'models').Model
    # --- 
    
    if arch == 'identity':
        model.to(device)
        return model
    
    if path is None:
        raise exceptions.ConfigurationError("Model configuration 'path' is not set. 'path' must be set when arch is not 'identity'.")
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    with open(path+"/checkpoint.pth", 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")[state_dict_key]
        
    if extract_method is None:
        pass
    elif extract_method == 'dpp':
        new_state_dict = {}
        
        prefix = 'module'
        for k, v in state_dict.items():
            assert k.startswith(prefix)
            new_state_dict[k[len(prefix)+1:]] = v

        state_dict = new_state_dict

    elif extract_method == 'moco':
        new_state_dict = {}

        encoder_module = 'module.encoder_q'
        old_fc_module = 'module.encoder_q.fc'

        for k, v in state_dict.items():
            if k.startswith(encoder_module) and not k.startswith(old_fc_module):
                new_state_dict[k[len(encoder_module)+1:]] = v

        state_dict = new_state_dict
    else:
        raise NotImplemenetedError()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    model.to(device)
    
    return model

def get_custom_model(arch, path=None, epoch=None, **kwargs):
    path = f"{env_config['model_base_path']}/{path}"
    if epoch is not None:
        path = f'{path}/{epoch:04d}.pth.tar'

    return _get_custom_model(arch, path=path, **kwargs)
