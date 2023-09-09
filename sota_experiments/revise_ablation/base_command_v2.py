import subprocess

features_list = ['object_i3d_feature', 'bbox_CLIP', 'geometric_feature',
                'object_semantic_embeddings', 'object_centric_shape_feats']

relative_features_list = ['relative_spatial_feature', 'interaction_bbox_CLIP', 
                        'interaction_centric_shape_feats']

# List of arguments to replace the last argument
arg_list = features_list + relative_features_list

gpus = [0,1,2,3,0,1,2,3]

base_command = "python executor/main_strat_custom_log.py --config configs/gpnn_ablation.yaml --ablated_feat "

# from tqdm import tqdm as tqdm

for arg in zip(arg_list, gpus):
    full_command = base_command + arg[0] + " --gpu " + str(arg[1])
    print(f"Executing: {full_command}")
    # subprocess.run(full_command, shell=True)
