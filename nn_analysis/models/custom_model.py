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

import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision.models
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# -- 

def _get_custom_model(arch, path=None, extract_method=None, model_kwargs={}, device='cpu', state_dict_key='state_dict'):
    archs_dict = {k: v for k, v in archs.__dict__.items() if not k.startswith("__") and callable(v) and k.islower()}
    #arch = 'factorize_avgpool_equivariant_all_bn_inj_v1'
    model = archs_dict[arch](**model_kwargs)
    
    print('arch: ', arch)
     
    
    if arch == 'identity':
        model.to(device)
        return model
    
    if path is None:
        raise exceptions.ConfigurationError("Model configuration 'path' is not set. 'path' must be set when arch is not 'identity'.")
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    print(path+"/checkpoint.pth")
    with open(path+"/checkpoint.pth.tar", 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")[state_dict_key]
        # --
        #model = torch.nn.DataParallel(model).cuda()
        """ckpt = torch.load(f, map_location="cpu")
        #print('cktp: ', ckpt.keys())
        #state_dict = model.load_state_dict(ckpt["model"])
        state_dict = ckpt["model"]
        #new_state_dict = {}
        #model = model.module
        
        new_state_dict = {}
        
        prefix = 'module'
        for k, v in state_dict.items():
            assert k.startswith(prefix)
            new_state_dict[k[len(prefix)+1:]] = v

        state_dict = new_state_dict
        
        #state_dict = model.load_state_dict(ckpt["model"])
        #state_dict = model.load_state_dict(state_d)"""
        #print(state_dict)
        print('model------')
        print(model)
        # -- 
        
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
