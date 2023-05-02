import numpy as np
import matplotlib.pyplot as plt

import plots
import config
import constants

config.results_file = constants.RESULTS_PATH


def save_results(model_names, metrics, layers, epochs, filename):
    results = {}
    for model_name in model_names:
        results[model_name] = plots.query_results(
            config.results_file,
            metrics,
            suppress=False,
            model_name=model_name,
            epoch=epochs,
            layer=layers,
        )
    plots.save_results(results, filename)
    
epoch = 30
model_names = [
    #'original_v2',
    'equivariant_all_bn_v1_v2',
#     'factorize_avgpool_equivariant_all_bn_v4',
#     'factorize_avgpool_equivariant_all_bn_v5',
]
metrics = config.metrics

layers = np.arange(17)
epochs = [epoch-1]
save_results(model_names, metrics, layers, epochs, f'{epoch}eps_layer-wise_results.pkl')

layers = [15,16]
epochs = np.arange(epoch)
save_results(model_names, metrics, layers, epochs, f'{epoch}eps_epoch-wise_results.pkl')
