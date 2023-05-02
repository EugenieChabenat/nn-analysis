__all__ = ['n_samples', 'save_acts_config', 'neural_fits_config', 'factorization_config', 'decoding_config', 'generalization_config', 'dimensionality_config', 'sparsity_config', 'rdm_config', 'rdm_2_config', 'results_file_config']

n_samples = 10000

save_acts_config = {
    'svd_solver': 'auto',
    'pca_configs': [{
            'dataset_configs': [('hk2', 'generalize', 0), ('hk2', 'generalize', 1)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[1] - arrs[0],
        }, {
            'dataset_configs': [('rust', 'neural_fits', 0)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('hvm', 'neural_fits', 0)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('hk2', 'decoding', 0)],
            'n_pcs': 1200,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('hk2', 'decoding', 0), ('hk2', 'decoding', 1)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[1] - arrs[0],
        }, {
            'dataset_configs': [('pseudo_hvm', 'factorize_obj_motion', 0)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('pseudo_hvm', 'factorize_background', 0)],
            'n_pcs': 300,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('pseudo_hvm', 'factorize_crop_10_jack', 0)],
            'n_pcs': 500,
            'transform': lambda arrs: arrs[0],
        }, {
            'dataset_configs': [('pseudo_hvm', 'factorize_color_10_jack', 0)],
            'n_pcs': 500,
            'transform': lambda arrs: arrs[0],
        },
    ]
}

neural_fits_config = {
    'dataset_configs': [
        ('hvm', 'hvm-neural_fits', 300, 300),
        ('rust', 'rust-neural_fits', 300, 150),
    ],
    'n_splits': 5,
    'test_size': 0.2,
    'cv_alpha_log_steps': True,
    'cv_alpha_min': 1.0e-2,
    'cv_alpha_max': 1.0e6,
    'cv_alpha_num_steps': 9,
    'alpha_per_target': False,
}

factorization_config = {
    'dataset_configs': [
        ('pseudo_hvm-factorize_obj_motion', 300, 0.9),
        ('pseudo_hvm-factorize_background', 300, 0.9),
        ('pseudo_hvm-factorize_crop_10_jack', 500, 0.9),
        ('pseudo_hvm-factorize_color_10_jack', 500, 0.9),
    ],
}

decoding_config = {
    'n_pcs': 300,
    'dataset_configs': [
        ('hk2-decoding', f'1200_pcs_{n_samples}_samples_0', ['obj_class'], 1.0e-4, 1.0e4),
        ('hk2-decoding', f'0300_pcs_{n_samples}_samples_0-{n_samples}_samples_1', ['cam_pos_x', 'cam_pos_y', 'cam_scale', 'brightness', 'contrast', 'saturation', 'hue'], 1.0e-2, 1.0e6),
        ('hvm-neural_fits', f'0300_pcs_{n_samples}_samples_0', ['obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z'], 1.0e-2, 1.0e6),
    ],
    'clf_type': 'svc',
    'stratify': True,
    'test_size': 0.2,
    'cv_regs_num_steps': 9,
    'cv': 5,
}

generalization_config = {
    'n_pcs': 300, 
    'dataset_name': 'hk2-generalize',
    'version': f'0300_pcs_{n_samples}_samples_0-{n_samples}_samples_1',
    'cv_regs_min': 1.0e-2,
    'cv_regs_max': 1.0e6,
    'cv_regs_num_steps': 9,
    'train_size': 0.125,
    'test_size': 0.25,
    'metrics': ['ccg_r2', 'parallelism'],
#     'metrics': ['parallelism'],
#     'metrics': ['ccg_r2'],
}

dimensionality_config = {
    'dataset_names': [
        'hk2-decoding',
#         'hvm-neural_fits',
    ],
    'version': f'1200_pcs_{n_samples}_samples_0',
}

sparsity_config = {
    'dataset_configs': [
        ('hk2', 'decoding', 0),
    ],
    'metrics': ['population', 'trial'],
    'zero_one': True,
}

rdm_config = {
    'rdm_similarity_metric': 'pearson',
    'rdm_comparison_metric': 'spearman',
    'n_splits': 3,
    'dataset_configs': [
        ('hvm', 'hvm-neural_fits', 300, 300, 500),
        ('rust', 'rust-neural_fits', 300, 150, 500),
    ],
}

rdm_2_config = {
    'rdm_similarity_metric': 'pearson',
    'rdm_comparison_metric': 'spearman',
    'n_splits': 5,
    'dataset_configs': [
        ('hvm', 'hvm-neural_fits', 300, 300, 16),
    ],
}

results_file_config = {
    'ccg': ['data', 'meta_data'],
    'decoding': ['data', 'meta_data'],
    'factorization': ['data'],
    'dimensionality': ['data'],
    'sparsity': ['data'],
#'neural_fits': ['data'],
    'neural_fits': [''],
    'rdm_neural_fits': ['data'],
    'rdm_neural_fits_2': ['data'],
}
