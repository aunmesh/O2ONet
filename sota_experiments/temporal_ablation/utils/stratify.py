from sklearn.model_selection import StratifiedKFold
import time

def get_stratified_splits_indices(train_dataset, n_splits=5):
    """
    Get stratified train and validation indices based on 'num_obj' field in each dictionary of the train_dataset.

    Args:
    - train_dataset (list): List of dictionaries where each dictionary contains a 'num_obj' field.
    - n_splits (int): Number of stratified splits.

    Returns:
    - train_indices (list): List containing n_splits lists, each with training indices for that split.
    - val_indices (list): List containing n_splits lists, each with validation indices for that split.
    """

    start_time = time.time()  # Start time

    # Extract 'num_obj' values from each dictionary to be used for stratification
    y = [d['num_obj'] for d in train_dataset.dataset]
    X = list(range(len(train_dataset)))  # Placeholder, as StratifiedKFold only needs y for stratification

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    train_indices = []
    val_indices = []
    
    for train_idx, val_idx in skf.split(X, y):
        train_indices.append(train_idx.tolist())
        val_indices.append(val_idx.tolist())

    end_time = time.time()  # End time

    # Print the time taken
    # elapsed_time = end_time - start_time
    # print(f"Time taken for stratified splitting: {elapsed_time:.2f} seconds")
        
    return train_indices, val_indices

# # Example Usage:
# train_dataset = [{'num_obj': 3}, {'num_obj': 2}, {'num_obj': 3}, {'num_obj': 2}, {'num_obj': 1}, {'num_obj': 2}, {'num_obj': 3}, {'num_obj': 1}, {'num_obj': 3}]
# train_indices, val_indices = get_stratified_splits_indices(train_dataset)
# print(train_indices)
# print(val_indices)


class CrossValidationAggregator:
    def __init__(self):
        # Initialize the data storage structure
        self.all_data = {}

    def add_data(self, fold, epoch, data):
        """
        Add data for a given fold and epoch.
        """
        if fold not in self.all_data:
            self.all_data[fold] = {}
        self.all_data[fold][epoch] = data

    def aggregate(self):
        """
        Aggregate the data and return the best epoch's data for each fold based on 'mAP_all' key.
        """
        best_data_per_fold = {}
        aggregated_results = {}

        # Find the best epoch for each fold
        for fold in self.all_data:
            best_mAP = -float('inf')
            best_epoch = None

            # for epoch in self.all_data[fold]:
            #     if self.all_data[fold][epoch]['mAP_all'] > best_mAP:
            #         best_mAP = self.all_data[fold][epoch]['mAP_all']
            #         best_epoch = epoch

            for epoch in self.all_data[fold]:
                for key in self.all_data[fold][epoch]:
                    if 'mAP_all' in key:
                        if self.all_data[fold][epoch][key] > best_mAP:
                            best_mAP = self.all_data[fold][epoch][key]
                            best_epoch = epoch


            best_data_per_fold[fold] = self.all_data[fold][best_epoch]

        # Aggregate the data of the best epoch for each fold
        keys = best_data_per_fold[list(best_data_per_fold.keys())[0]].keys()
        for key in keys:
            aggregated_results['Agg_' + key] = sum([best_data_per_fold[fold][key] for fold in best_data_per_fold]) / len(best_data_per_fold)

        return aggregated_results


def modify_keys_for_fold(results_dict, fold_number):
    """Modify the keys of results_dict by appending the fold_number."""
    modified_dict = {}
    for key, value in results_dict.items():
        modified_key = f"fold_{fold_number}_{key}"
        modified_dict[modified_key] = value
    return modified_dict

