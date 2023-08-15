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


dict_color = {
    # new architectures
    "inj_v1_evaluate_IT":  ["magenta", '-'],
    "inj_v2_evaluate_IT":  ["forestgreen", '-'],

    # no projector controls 
    "noprojector_control_v1":  ["black", '-'], 
    "noprojector_control_v2":  ["black", '-'], 
    "noprojector_control_v4":  ["black", '-'], 
    "noprojector_control_IT":  ["black", '-'], 
    
    # no projector linear 
    "noprojector_linear_v1":  ["brown", '-'], 
    "noprojector_linear_v2":  ["brown", '-'], 
    "noprojector_linear_v4":  ["brown", '-'], 
    "noprojector_linear_IT":  ["brown", '-'], 
    
    #'injection_conv_subset_v1_proj':["black", '-'],
    #"noprojector_linear_v1_nm3":  ["brown", '--'], 
    
    # no projector conv
    "noprojector_conv_v1": ["gold", '-'], 
    "noprojector_conv_v2": ["gold", '-'], 
    "noprojector_conv_v4": ["gold", '-'], 
    "noprojector_conv_IT": ["gold", '-'], 
    
    # random injection 
    "injection_v1_af" : ["orange", ':'], #, '-'],
    "injection_v2_af": ["orange", ':'], #, '-'],
    "injection_v4_af": ["orange", ':'], #, '-'],
    "injection_IT_af": ["orange", ':'], #, '-'],
    
    # convolution injection
    "injection_conv_v1_af": ["lightblue", ':'], #, '-'],
    "injection_conv_v2_af": ["lightblue", ':'], #, '-'],
    "injection_conv_v4_af": ["lightblue", ':'], #, '-'],
    "injection_conv_IT_af": ["lightblue", ':'], #, '-'],
    
    #"injection_conv_v1_af": ["red", '-'], 
    #"injection_conv_v2_af": ["blue", '-'], 
    #"injection_conv_v4_af": ["orange", '-'], 
    #"injection_conv_IT_af": ["green", '-'], 
       
    # unfreeze convolution injection 
    "unfreeze_injection_v1_af": ["green", '-'], 
    "unfreeze_injection_v2_af": ["green", '-'], 
    "unfreeze_injection_v4_af": ["green", '-'], 
    "unfreeze_injection_IT_af": ["green", '-'], 
    
    #"unfreeze_injection_v1_af": ["red", '-'], 
    #"unfreeze_injection_v2_af": ["blue", '-'], 
    #"unfreeze_injection_v4_af": ["orange", '-'], 
    #"unfreeze_injection_IT_af": ["green", '-'], 

    # subset injection 
    "subset_injection_v1": ["blue", '-'], 
    "subset_injection_v2": ["blue", '-'], 
    "subset_injection_v4": ["blue", '-'], 
    "subset_injection_IT": ["blue", '-'],

    # conv subset injection 
    "injection_conv_subset_v1": ["lime", '-'], 
    "injection_conv_subset_v2": ["lime", '-'], 
    "injection_conv_subset_v4": ["lime", '-'], 
    "injection_conv_subset_IT": ["lime", '-'],

    # separate injection 
    "injection_separate_v1": ["purple", '-'], 
    "injection_separate_v2": ["purple", '-'], 
    "injection_separate_v4": ["purple", '-'], 
    "injection_separate_IT": ["purple", '-'], 
  
    # conv subset injection 
    "injection_conv_subset_v1": ["lime", '-'], 
    "injection_conv_subset_v2": ["lime", '-'], 
    "injection_conv_subset_v4": ["lime", '-'], 
    "injection_conv_subset_IT": ["lime", '-'],

    # control models 
    "v1_no_injection": ["red", '--'], 
    "v2_no_injection": ["red", '--'], 
    "v4_no_injection": ["red", '--'], 
    "IT_no_injection": ["red", '--'], 

    "resnet50_untrained": ["pink", '--'], 
    "barlow_twins_50epochs": ["grey", '--'], 
    "barlow_fact_no_injection": ["black", '--']
}
# --- 
epoch = 29
layers = np.arange(2)
layers =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]#, 21, 22, 23, 24, 25, 26, 27]

#layers =[ 18, 19, 20]#, 21, 22, 23, 24, 25, 26, 27]
#layers = [ 21, 22, 23, 24, 25, 26, 27]

one_layer = 0
metric = ["curve", 1]

model_names = [
    # new architectures 
    "inj_v1_evaluate_IT", "inj_v2_evaluate_IT", 
    
    # control no projector
    #"noprojector_control_v1",  #"noprojector_control_v2", "noprojector_control_v4", "noprojector_control_IT",
     
    # random linear no projector
    #"noprojector_linear_v1", "noprojector_linear_v2", "noprojector_linear_v4", "noprojector_linear_IT", 
    
    # random convolution no projector 
    #"noprojector_conv_v1", "noprojector_conv_v2", "noprojector_conv_v4", "noprojector_conv_IT", 
    
    # random injection models  
    #"injection_v1_af","injection_v2_af", "injection_v4_af", "injection_IT_af",
    
    # convolution injection models 
    #"injection_conv_v1_af", "injection_conv_v2_af", "injection_conv_v4_af", "injection_conv_IT_af", 
    
    # unfreeze convolution injection models 
    #"unfreeze_injection_v1_af", "unfreeze_injection_v2_af", "unfreeze_injection_v4_af", "unfreeze_injection_IT_af", 

    # subset 
    #"subset_injection_v1", "subset_injection_v2", "subset_injection_v4", "subset_injection_IT",

    # conv subset injection 
    #"injection_conv_subset_v1", "injection_conv_subset_v2", "injection_conv_subset_v4", "injection_conv_subset_IT",

    # separate learning of weights 
    #"injection_separate_v1", "injection_separate_v2", "injection_separate_v4", "injection_separate_IT",
    
    # control models 
    #"v1_no_injection", "v2_no_injection", "v4_no_injection", "IT_no_injection", 

    "resnet50_untrained", 
    "barlow_twins_50epochs", 
]
metric_types = ["x_cam_trans", "y_cam_trans", "z_cam_trans", "x_cam_rot", "y_cam_rot", 'x_cam_pan', 'yz_cam_pan']

nb_metrics = len(metric_types)


# ----- 
# computing prediction loss
# ----- 

path = '/mnt/smb/locker/issa-locker/users/Eugénie/models/checkpoints/barlowtwins/multiplicative_separate_v2_v1/stats.txt'
list_lines = []
with open(path, 'r') as f:
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
  #print(line)
  if line["epoch"] <= 29:
    epochs.append(line["epoch"])
    steps.append(line["step"])
    losses.append(line["loss"])

  #while line["epoch"] <= 29: 
    if line["epoch"] == current_e: 
      ind +=1 
    else: 
      if inds: 
        inds.append(ind+inds[-1])
      else : 
        inds.append(ind)
      ind = 0 
      current_e =line["epoch"]
print(current_e) 
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#labels = np.arange(0, 29, step=1)
new_labels = []
new_inds= []  
i = 0
for element in labels:   
  if element == 0 or element %5 ==0 or element ==29: 
    new_labels.append(element)
    new_inds.append(inds[i])
  i +=1
    
print('last loss: ', losses[-1])
print('loss: ', losses)

 # ------


#fig, axes = pt.core.subplots(2, 2, size=(10, 10), sharex=True)
plt.figure(figsize=(15,15))
average_identity_scores = []
 
for i, metric_type in enumerate(metric_types): 
  scores = [load_data(metric, 'identity', None, 0)[metric_type] for layer in layers]
  if average_identity_scores: 
      average_identity_scores = [sum(x) for x in zip(scores, average_identity_scores)]
  else: 
      average_identity_scores = scores
average_identity_scores= [x/nb_metrics for x in average_identity_scores]
scores_id = [i * 180 for i in average_identity_scores]
print('pixels: ', np.mean(scores_id))

for model_name in model_names: 
  average_scores = []
  #print(model_name)
  for i, metric_type in enumerate(metric_types): 
    scores = [load_data(metric, model_name, epoch, layer)[metric_type] for layer in layers]
    if average_scores: 
      average_scores = [sum(x) for x in zip(scores, average_scores)]
    else: 
      average_scores = scores
        
  average_scores = [x/nb_metrics for x in average_scores]
  scores = [i * 180 for i in average_scores]
  #print(model_name)
  #print(np.mean(scores))
  
    
  plt.plot(layers, average_scores, label=dict_model_names[model_name], color = dict_color[model_name][0], ls = dict_color[model_name][1])
  plt.plot(layers, average_identity_scores, label='identity', color = 'grey')


#axes[key,i].set_title(dict_metric_names[metric_type], fontsize=18)#, fontsize =60)
plt.xticks([0, 3, 6, 10, 16, 19, 20], labels=['1st convolution', 'maxpool', 'v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, fontsize=14)
#plt.xticklabels(['1st convolution', 'maxpool', 'v1 injection', 'v2 injection', 'v4 injection', 'IT injection', 'avgpool'], rotation=45, ha='right', fontsize=16)#, fontsize=60)

plt.text(4.5, 0.95, "Block V1", ha="center", va="center", size=14)#, size=60)
plt.text(8, 0.95, "Block V2", ha="center", va="center", size=14)#, size=60)
plt.text(13, 0.95, "Block V4", ha="center", va="center", size=14)#, size=60)
plt.text(17.5, 0.95, "Block IT", ha="center", va="center", size=14)#, size=60)
#axes[0,i].text(23.5, 0.95, "Projector", ha="center", va="center", size=12)#, size=10)
plt.ylim(0.0, 1.)
plt.tick_params(axis='y', labelsize=14)
#axes[0,i].legend()#loc='center left')

#plt.legend(loc='lower center', bbox_to_anchor=(1.25, 0.5), fontsize=18)#, fontsize=60)
#fig.supxlabel('layers')#, fontsize=60)
#fig.supylabel('decode')#, fontsize=60)
#fig.tight_layout()
plt.xlabel('layers', fontsize=14)
plt.ylabel('average curvature score', fontsize=14)
#plt.title('Injection and Evaluation at IT', fontsize=20)
plt.title('Average Curvature score', fontsize=20)
plt.show()
plt.savefig('/mnt/smb/locker/issa-locker/users/Eugénie/nn-analysis/avg-curve-v12_pres.png')
plt.savefig('/home/ec3731/issa_analysis/nn-analysis/1-avg-curve-v12_pres.png')
