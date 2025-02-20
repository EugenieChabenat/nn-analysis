__all__ = ['metrics', 'metric_configs']

# list of commonly used metrics
metrics = [
    *[f'neural fits - {region} ({base_dataset})' for region in ['V4', 'IT'] for base_dataset in ['hvm', 'rust']],
    *[f'RDM - {region} ({base_dataset})' for region in ['V4', 'IT'] for base_dataset in ['hvm', 'rust']],
    *[f'RDM2 - {region} ({base_dataset})' for region in ['V4', 'IT'] for base_dataset in ['hvm']],
    *[f'decode - {var}' for var in [
        'obj_class',
        'cam_pos',
#         'cam_pos_x',
#         'cam_pos_y',
        'cam_scale',
        'obj_pos',
#         'obj_pos_x',
#         'obj_pos_y',
        'obj_scale',
        'obj_pose',
#         'obj_pose_x',
#         'obj_pose_y',
#         'obj_pose_z',
        'lighting',
        'color',
#         'brightness',
#         'contrast',
#         'saturation',
#         'hue',
    ]],
    *[f'{metric} - {var}' for metric in ['fact', 'ss_inv', 'inv'] for var in ['crop', 'lighting', 'obj_motion', 'background']],
    *[f'{metric} - {var}' for metric in [
#         'ccg_cc',
        'ccg_r2',
        'parallelism'
    ] for var in [
        'cam_pos',
#         'cam_pos_x',
#         'cam_pos_y',
        'cam_scale',
        'lighting',
        'color',
#         'brightness',
#         'contrast',
#         'saturation',
#         'hue',
    ]],
    *['dimensionality', 'sparsity'],
]

metric_configs = {
    **{f'neural fits - {region} ({base_dataset})': {
        'where': '/neural_fits',
        'key': 'score',
        'dataset_name': f'{base_dataset}-neural_fits',
        'version': '10000_samples_0',
        'region': region,
        'n_pcs': {'hvm': 300, 'rust': 150}[base_dataset],
        'cv_alpha_max': 1.0e6,
        'cv_alpha_min': 1.0e-2,
#         'cv_k_folds': 5,
    } for region in ['V4', 'IT'] for base_dataset in ['hvm', 'rust']},
    **{f'RDM - {region} ({base_dataset})': {
        'where': '/rdm_neural_fits',
        'key': 'score',
        'dataset_name': f'{base_dataset}-neural_fits',
        'version': '10000_samples_0',
        'region': region,
        'n_pcs': {'hvm': 300, 'rust': 150}[base_dataset],
        'n_images': 500,
        'rdm_similarity_metric': 'pearson',
        'rdm_comparison_metric': 'spearman',
    } for region in ['V4', 'IT'] for base_dataset in ['hvm', 'rust']},
    **{f'RDM2 - {region} ({base_dataset})': {
        'where': '/rdm_neural_fits_2',
        'key': 'score',
        'dataset_name': f'{base_dataset}-neural_fits',
        'version': '10000_samples_0',
        'region': region,
        'n_pcs': 300,
        'n_partitions': 16,
        'n_splits': 5,
        'rdm_similarity_metric': 'pearson',
        'rdm_comparison_metric': 'spearman',
    } for region in ['V4', 'IT'] for base_dataset in ['hvm']},
    'decode - obj_class': {
        'where': '/decoding',
        'key': 'obj_class',
        'dataset_name': 'hk2-decoding',
        'version': '1200_pcs_10000_samples_0',
        'n_pcs': 300,
        'cv_regs_max': 1.0e4,
        'cv_regs_min': 1.0e-4,
        'cv_regs_num_steps': 9,
        'cv': 5,
        'clf_type': 'svc',
        'stratify': True,
    },
    **{f'decode - {var}': {
        'where': '/decoding',
        'key': var,
        'dataset_name': 'hk2-decoding',
#         'version': '1200_pcs_100000_samples_0-100000_samples_1',
        'version': '0300_pcs_10000_samples_0-10000_samples_1',
        'n_pcs': 300,
        'cv_regs_max': 1.0e6,
        'cv_regs_min': 1.0e-2,
        'cv_regs_num_steps': 9,
        'cv': 5,
    } for var in ['cam_pos', 'cam_pos_x', 'cam_pos_y', 'cam_scale', 'lighting', 'color', 'brightness', 'contrast', 'saturation', 'hue']},
    **{f'decode - {var}': {
        'where': '/decoding',
        'key': var,
        'dataset_name': 'hvm-neural_fits',
#         'version': '1200_pcs_100000_samples_0',
        'version': '0300_pcs_10000_samples_0',
        'n_pcs': 300,
        'cv_regs_max': 1.0e6,
        'cv_regs_min': 1.0e-2,
        'cv_regs_num_steps': 9,
        'cv': 5,
    } for var in ['obj_pos', 'obj_pose', 'obj_pos_x', 'obj_pos_y', 'obj_scale', 'obj_pose_x', 'obj_pose_y', 'obj_pose_z']},
    **{f'{metric} - {var}': {
        'where': '/factorization',
        'key': metric,
        'dataset_name': f'pseudo_hvm-factorize_{var}',
#         'dataset_name': f'pseudo_hvm-factorize_{var}_jack',
        'version': '0300_pcs_10000_samples_0',
#         'version': '-001_pcs_10000_samples_0',
        'n_pcs': 300,
#         'n_pcs': -1,
        'threshold': 0.9,
    } for metric in ['fact', 'ss_inv', 'inv'] for var in ['obj_motion', 'background']},
    **{f'{metric} - {var}': {
        'where': '/factorization',
        'key': metric,
#         'dataset_name': f'pseudo_hvm-factorize_{var}_20',
        'dataset_name': f'pseudo_hvm-factorize_{var}_10_jack',
        'version': '0500_pcs_10000_samples_0',
#         'version': '-001_pcs_10000_samples_0',
        'n_pcs': 500,
#         'n_pcs': -1,
        'threshold': 0.9,
    } for metric in ['fact', 'ss_inv', 'inv'] for var in ['crop', 'color']},
    **{f'{metric} - lighting': {
        'where': '/factorization',
        'key': metric,
#         'dataset_name': f'pseudo_hvm-factorize_color_20',
        'dataset_name': f'pseudo_hvm-factorize_color_10_jack',
        'version': '0500_pcs_10000_samples_0',
#         'version': '-001_pcs_10000_samples_0',
        'n_pcs': 500,
#         'n_pcs': -1,
        'threshold': 0.9,
    } for metric in ['fact', 'ss_inv', 'inv']}, # same as color factorization scores, just different name
    **{f'{metric} - {var}': {
        'where': '/ccg',
        'key': f'{metric}_{var}',
        'dataset_name': f'hk2-generalize',
#         'version': '1200_pcs_100000_samples_0-100000_samples_1',
        'version': '0300_pcs_10000_samples_0-10000_samples_1',
        'n_pcs': 300,
        'train_size': 0.125,
        'test_size': 0.25,
        'cv_regs_min': 0.01,
        'cv_regs_max': 1000000,
        'cv_regs_num_steps': 9,
    } for metric in ['ccg_cc', 'ccg_r2', 'parallelism'] for var in ['cam_pos', 'cam_pos_x', 'cam_pos_y', 'cam_scale', 'lighting', 'color', 'brightness', 'contrast', 'saturation', 'hue']},
    'dimensionality': {
        'where': '/dimensionality',
        'key': 'pr',
        'dataset_name': 'hk2-decoding',
#         'version': '1200_pcs_100000_samples_0',
        'version': '1200_pcs_10000_samples_0',
    },
    'sparsity': {
        'where': '/sparsity',
        'key': 'population',
        'dataset_name': 'hk2-decoding',
#         'version': '100000_samples_0',
        'version': '10000_samples_0',
        'zero_one': True,
    },
}
