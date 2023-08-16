import numpy as np
import matplotlib.pyplot as plt
import torch
from nn_analysis import metrics as me
from nn_analysis import utils
from nn_analysis import plot as pt

import json
import pickle

import os

def load_data(metric, model_name, epoch, layers):
    layer_names = utils.get_layer_names(model_name, layers)
    if isinstance(layer_names, list):
        return [me.utils.load_data(model_name, epoch, layer_name, metric[0], metric[1]) for layer_name in layer_names]
    else:
        return me.utils.load_data(model_name, epoch, layer_names, metric[0], metric[1])

  
# --- 


colors = ['red', 'blue', 'green', 'black']
for i in range(10): 
    colors.append('red')
    colors.append('blue')
    colors.append('green')
    colors.append('black')

print(colors)

epoch = 29
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
metric = ["curve", 1]

model_names = [
    # new architectures 
    #"inj_v1_evaluate_IT", "inj_v2_evaluate_IT", 
    # control no projector
    "no_projector_linear_control_v1",  "no_projector_linear_control_v2", "no_projector_linear_control_v4", "no_projector_linear_control_IT",
    # random linear no projector
    "no_projector_linear_v1", "no_projector_linear_v2", "no_projector_linear_v4", #"no_projector_linear_IT", 
    # random convolution no projector 
    "no_projector_conv_v1", "no_projector_conv_v2", "no_projector_conv_v4", "no_projector_conv_IT", 
    # random injection models  
    "new_injection_v1", "new_injection_v2", "new_injection_v4", "new_injection_IT",
    # convolution injection models 
    "new_injection_conv_v1", "new_injection_conv_v2", "new_injection_conv_v4", #"new_injection_conv_IT", 
    # unfreeze convolution injection models 
    "unfreeze_injection_conv_v1", "unfreeze_injection_conv_v2", "unfreeze_injection_conv_v4", "unfreeze_injection_conv_IT", 
    # subset 
    "injection_subset_v1", "injection_subset_v2", "injection_subset_v4", "injection_subset_IT",

    # conv subset injection 
    "injection_conv_subset_v1", "injection_conv_subset_v2", "injection_conv_subset_v4", "injection_conv_subset_IT",

    # separate learning of weights 
    "injection_separate_v1", "injection_separate_v2", "injection_separate_v4", "injection_separate_IT",
    
    # control models 
    "v1_no_injection", "v2_no_injection", "v4_no_injection", "IT_no_injection", 

    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
]
metric_types = ["x_cam_trans", "y_cam_trans", "z_cam_trans", "x_cam_rot", "y_cam_rot", 'x_cam_pan', 'yz_cam_pan']

nb_metrics = len(metric_types)
all_losses = []
# ----- 
# computing prediction loss
# ----- 

path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/'

for model_name in model_names: 
    print(model_name)
    list_lines = []
    if model_name[-1] == '1': 
        complete_path = path+model_name+'/stats.txt'
    else: 
        complete_path = path+model_name+'_v1/stats.txt'
    with open(complete_path, 'r') as f:
      lines = f.readlines()
    
    list_lines = []
    
    for line in lines: 
      if line[0] == "{": 
        list_lines.append(json.loads(line))

    epochs = []
    steps = []
    losses = []
    inds = [0]
    current_e = 0 
    ind = 0 
    for line in list_lines: 
      if line["epoch"] <= 29:
        epochs.append(line["epoch"])
        steps.append(line["step"])
        losses.append(line["loss"])
        if line["epoch"] == current_e: 
          ind +=1 
        else: 
          if inds: 
            inds.append(ind+inds[-1])
          else : 
            inds.append(ind)
          ind = 0 
          current_e =line["epoch"]
    all_losses.append(losses[-1])

 # ------
model_names = [
    # new architectures 
    #"inj_v1_evaluate_IT", "inj_v2_evaluate_IT",
    # control no projector
    "noprojector_control_v1",  "noprojector_control_v2", "noprojector_control_v4", "noprojector_control_IT",
    # random linear no projector
    "noprojector_linear_v1", "noprojector_linear_v2", "noprojector_linear_v4", #"noprojector_linear_IT", 
    
    # random convolution no projector 
    "noprojector_conv_v1", "noprojector_conv_v2", "noprojector_conv_v4", "noprojector_conv_IT", 
    
    # random injection models  
    "injection_v1_af", "injection_v2_af", "injection_v4_af", "injection_IT_af",
    
    # convolution injection models 
    "injection_conv_v1_af", "injection_conv_v2_af", "injection_conv_v4_af", #"injection_conv_IT_af", 
    
    # unfreeze convolution injection models 
    "unfreeze_injection_v1_af", "unfreeze_injection_v2_af", "unfreeze_injection_v4_af", "unfreeze_injection_IT_af", 

    # subset 
    "subset_injection_v1", "subset_injection_v2", "subset_injection_v4", "subset_injection_IT",

    # conv subset injection 
    "injection_conv_subset_v1", "injection_conv_subset_v2", "injection_conv_subset_v4", "injection_conv_subset_IT",

    # separate learning of weights 
    "injection_separate_v1", "injection_separate_v2", "injection_separate_v4", "injection_separate_IT",
    
    # control models 
    "v1_no_injection", "v2_no_injection", "v4_no_injection", "IT_no_injection", 

    #"resnet50_untrained", 
    #"barlow_twins_50epochs", 
]
average_identity_scores = []
all_scores = []
for model_name in model_names: 
  average_scores = []
  for i, metric_type in enumerate(metric_types): 
    scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
    if average_scores: 
      average_scores = [sum(x) for x in zip(scores, average_scores)]
    else: 
      average_scores = scores
        
  average_scores = [x/nb_metrics for x in average_scores]
  scores = [i * 180 for i in average_scores]
  all_scores.append(np.mean(scores))


colors = ['red', 'blue', 'green', 'black']
for i in range(10): 
    colors.append(colors)
    
plt.figure(figsize=(15,15))
plt.scatter(all_losses, all_scores, c=colors )
plt.xlabel('prediction loss')
plt.ylabel('average straightening score')
plt.show()
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/scatter_loss_curve3.png')
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/scatter_loss_curve.png')
