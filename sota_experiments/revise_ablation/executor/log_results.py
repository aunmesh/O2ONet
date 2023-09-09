import wandb
import os

class wandb_logger:

    def __init__(self, config):

        self.wandb_log = 1

        proj_name = config['wandb_project_name']
        entity = config['wandb_entity']

        if 'run_id' not in config.keys():
            run_id = wandb.util.generate_id()
            wandb.init(project=proj_name, entity=entity, id=run_id, resume="allow", config=config)
            wandb.config.update({"run_id": run_id})
            config['run_id'] = run_id
            self.config = config

        else:
            run_id = config['run_id']
            wandb.init(project=proj_name, entity=entity, id=run_id, resume="allow", config=config)
            
            try:
                wandb.restore( os.path.join(config['model_saving_path'], run_id, 'checkpoint.pth') )
            except:
                pass
            
            self.config = config

    
    def watch(self, model):
        wandb.watch(model, log_freq=1)

    def log_dict(self, log_dict):

        wandb.log(log_dict)




import os
import datetime
import pickle



class CrossValidationLogger:
    def __init__(self, config):
        
        self.config = config
        
        self.num_folds = config['num_folds']
        self.log_dir = config['master_log_dir'] + '/' + config['ablated_model_name'] + '/'
        
        self.experiment_name = config['split_file_id']
                
        self.experiment_metadata = self.config
        
        self.num_folds = config['num_folds']
        self.fold_data = {f"fold_{i + 1}": {} for i in range(self.num_folds)}
        self.summary_metrics = {}
        
        self.save_path = os.path.join(self.log_dir, f"{self.experiment_name}_log.pkl")
        
        # Ensure log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_fold_start(self, fold_num, train_indices, val_indices):
        """Log the start of a fold."""
        fold_key = f"fold_{fold_num}"
        self.fold_data[fold_key]['train_indices'] = train_indices
        self.fold_data[fold_key]['val_indices'] = val_indices
        self.fold_data[fold_key]['start_time'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.fold_data[fold_key]['metrics'] = []

    def log_fold_end(self, fold_num):
        """Log the end of a fold."""
        fold_key = f"fold_{fold_num}"
        self.fold_data[fold_key]['end_time'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


    def log_fold_metrics(self, fold_num, metric_dict):
        """Log metrics for a fold."""
        fold_key = f"fold_{fold_num}"
        self.fold_data[fold_key]['metrics'].append(metric_dict)



    def data_exists(self):
        """Check if the experiment data already exists on disk."""
        return os.path.exists(self.save_path)


    def load(self):
        """Load the experiment metadata and fold data from the file on disk."""


        if os.path.exists(self.save_path):

            with open(self.save_path, 'rb') as file:
                loaded_data = pickle.load(file)

            # Loading data into class variables
            self.experiment_metadata = loaded_data.get('metadata', {})
            self.fold_data = loaded_data.get('fold_data', {})
            self.summary_metrics = loaded_data.get('summary_metrics', {})
            
            return True  # Return True to indicate successful loading
        else:
            return False  # Return False to indicate data does not exist

    def last_completed_fold(self):
        """Returns the number of the last completed fold or None if no fold was completed."""
        for fold_num in range(self.num_folds, 0, -1):  # Starting from the last fold and going backwards
            fold_key = f"fold_{fold_num}"
            if fold_key in self.fold_data and 'end_time' in self.fold_data[fold_key]:
                return fold_num
        return None


    def save(self):
        """Save the experiment metadata and fold data to one file on disk."""
        
        # Combining metadata and fold data into one dictionary
        data_to_save = {
            'metadata': self.experiment_metadata,
            'fold_data': self.fold_data,
            'summary_metrics': self.summary_metrics
        }

        with open(self.save_path, 'wb') as file:
            pickle.dump(data_to_save, file)
        temp = self.last_completed_fold()
        if temp == None:
            temp = 1
        print(f"Saved experiment data for fold " + str(temp) )


    def aggregate(self):
        """
        Aggregate the data and return the best epoch's data for each fold based on 'mAP_all' key.
        """
        best_data_per_fold = {}
        aggregated_results = {}

        # Find the best epoch for each fold
        for fold in self.fold_data:
            # best_mAP = -float('inf')
            # best_epoch = None

            # # Assuming the structure of fold_data[fold] is:
            # # {epoch_1: {...}, epoch_2: {...}, ...}
            
            # for epoch in self.fold_data[fold]:
            #     if 'metrics' not in self.fold_data[fold][epoch]:  # Skip if metrics key not present
            #         continue
                
            #     for metric_key in self.fold_data[fold][epoch]['metrics']:
            #         if 'val_mAP_all' in metric_key:
            #             if self.fold_data[fold][epoch]['metrics'][metric_key] > best_mAP:
            #                 best_mAP = self.fold_data[fold][epoch]['metrics'][metric_key]
            #                 best_epoch = epoch
            # if best_epoch == None:
            #     print("ERROR: No best epoch found for fold " + str(fold) )
            best_data_per_fold[fold] = self.fold_data[fold]['metrics'][8]
        
        # Aggregate the data of the best epoch for each fold
        keys = best_data_per_fold[list(best_data_per_fold.keys())[0]].keys()

        for key in keys:
            temp_key = key[7:]
            # print("flag", key, temp_key)
            aggregated_results['Agg_' + temp_key] = sum([best_data_per_fold[fold]['fold_' + str(i) + '_' + temp_key] for i, fold in enumerate(best_data_per_fold)]) / len(best_data_per_fold)

        self.summary_metrics = aggregated_results