import subprocess
import time

features_list = ['object_i3d_feature', 'bbox_CLIP', 'geometric_feature',
                'object_semantic_embeddings', 'object_centric_shape_feats']

relative_features_list = ['relative_spatial_feature', 'interaction_bbox_CLIP', 
                        'interaction_centric_shape_feats']

arg_list = features_list + relative_features_list
gpus = [0,1,2,3,0,1,2,3]
base_command = "python executor/main_strat_custom_log.py --config configs/gpnn_ablation.yaml --ablated_feat "

MAX_RETRIES = 10

processes = {base_command + arg[0] + " --gpu " + str(arg[1]): {'process': None, 'retries': 0} for arg in zip(arg_list, gpus)}

print("Created processes dictionary")

# Initial launch of all processes without sleep
for cmd, data in processes.items():
    print(f"Launching: {cmd}")
    data['process'] = subprocess.Popen(cmd, shell=True)
    time.sleep(5)  # Sleep for 10 seconds between launches

print("Launched all processes")
counter = 0

while processes:
    
    print("Checking processes " + str(counter))
    counter+=1
    
    time.sleep(45)  # Sleep for 45 seconds between checks

    # Check the processes
    for cmd, data in list(processes.items()):  # Convert dict items to list to safely modify the dictionary while iterating
        if data['process'].poll() is not None:  # Process has finished
            if data['process'].returncode != 0:  # Error in execution
                if data['retries'] < MAX_RETRIES:
                    print(f"Retrying: {cmd}")
                    data['retries'] += 1
                    data['process'] = subprocess.Popen(cmd, shell=True)
                    time.sleep(2)  # Sleep for 10 seconds between launches                    
                else:
                    print(f"Command {cmd} failed after maximum retries.")
                    del processes[cmd]                    
            else:
                print(f"Command {cmd} completed successfully!")
                del processes[cmd]

print("All processes attempted!")
