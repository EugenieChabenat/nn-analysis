import numpy as np

import constants

__all__ = ['model_configs']

model_configs = [
        {
            'model_names': [
#                 'original_v1',
#                 'original_v2',
#                'original_v3',
#                 'equivariant_pos_v2',
#                 'equivariant_pos_v3',
#                 'equivariant_pos_bn_v2',
#                 'equivariant_scale_v1',
#                 'equivariant_scale_v2',
#                 'equivariant_scale_v3',
#                 'equivariant_scale_bn_v2',
#                 'equivariant_color_v1',
#                 'equivariant_color_bn_v2',
#                 'equivariant_all_v1',
#                 'equivariant_all_bn_v2',
#                 'factorize_avgpool_equivariant_all_bn_v4',
                 'factorize_avgpool_equivariant_all_bn_v5',
#                 'factorize_avgpool_equivariant_all_bn_2_v1',
#                 'factorize_avgpool_decode_avgpool_v2',
#                 'factorize_avgpool_v1',
#                 'factorize_avgpool_decode_avgpool_v3',
#                 'decode_avgpool_v1',
            ],
            'epochs': [20],
            'layers': np.arange(17),
        },
        {
            'model_names': [
#                 'original_v1',
#                 'original_v2',
                   #'original_v3',
#                 'equivariant_pos_v2',
#                 'equivariant_pos_v3',
#                 'equivariant_pos_bn_v2',
#                 'equivariant_scale_v1',
#                 'equivariant_scale_v2',
#                 'equivariant_scale_v3',
#                 'equivariant_scale_bn_v2',
#                 'equivariant_color_v1',
#                 'equivariant_color_bn_v2',
#                 'equivariant_all_v1',
#                 'equivariant_all_bn_v2',
#                 'factorize_avgpool_equivariant_all_bn_v4',
                 'factorize_avgpool_equivariant_all_bn_v5',
#                 'factorize_avgpool_equivariant_all_bn_2_v1',
#                 'factorize_avgpool_decode_avgpool_v2',
#                 'factorize_avgpool_v1',
#                 'factorize_avgpool_decode_avgpool_v3',
#                 'decode_avgpool_v1',
            ],
            'epochs': np.arange(19),
            'layers': [15,16],
        },
        {
            'model_names': [
#                 'original_v1',
#                 'original_v2', # done
#                'original_v3',
#                 'equivariant_pos_v2',
#                 'equivariant_pos_bn_v2', # done
#                 'equivariant_pos_v3',
#                 'equivariant_scale_v2', # done
#                 'equivariant_scale_v3',
#                 'equivariant_scale_bn_v2', # done
#                 'equivariant_color_v1', # done
#                 'equivariant_color_bn_v2', # done
#                 'equivariant_all_v1', # done
#                 'equivariant_all_bn_v2', # done
                 'factorize_avgpool_equivariant_all_bn_v5', # done
#                 'factorize_avgpool_equivariant_all_bn_2_v1', # done
#                 'factorize_avgpool_decode_avgpool_v2', # done
#                 'factorize_avgpool_v1', # done
#                 'factorize_avgpool_decode_avgpool_v3', # done
#                 'decode_avgpool_v1', # done
            ],
            'epochs': [49],
            'layers': np.arange(17),
        },
        {
            'model_names': [
#                 'original_v1',
#                 'original_v2', # done
#                'original_v3',
#                 'equivariant_pos_v2',
#                 'equivariant_pos_bn_v2', # done
#                 'equivariant_pos_v3',
#                 'equivariant_scale_v2', # done
#                 'equivariant_scale_v3',
#                 'equivariant_scale_bn_v2', # done
#                 'equivariant_color_v1', # done
#                 'equivariant_color_bn_v2', # done
#                 'equivariant_all_v1', # done
#                 'equivariant_all_bn_v2', # done
                 'factorize_avgpool_equivariant_all_bn_v5', # done
#                 'factorize_avgpool_equivariant_all_bn_2_v1', # done
#                 'factorize_avgpool_decode_avgpool_v2', # done
#                 'factorize_avgpool_v1', # done
#                 'factorize_avgpool_decode_avgpool_v3', # done
#                 'decode_avgpool_v1', # done
            ],
            'epochs': np.arange(20,49),
            'layers': [15,16],
        },
]
