from models import BarlowTwins
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Model-specific parameters')
    return parser

class Model(BarlowTwins):
    def __init__(self, args):
        super().__init__(args)
        
    def forward(self, *args, **kwargs):
        loss = super().forward(*args)
        loss_components = {'Contrastive Loss': loss.item()}
        return loss, loss_components
    
    def get_encoder(self):
        return self.backbone
