import os
import random
import math
class DatasetPreparer:

    def __init__(self, config):
        self.config = config

    def _read_paths(self, BASE_PATH, ending, X_path = None):
        paths = []
        if ending == '.tif':
            for root, dirs, files in os.walk(BASE_PATH):
                for file in sorted(files):
                    if file.endswith(ending): # should not be necessary
                        paths.append(os.path.join(root, file))

        elif ending == 'data.npy' or ending == 'masks.npy' or ending == 'products.txt':
            #Stores the number of times a sentinel image/mask/time series info should be added to path list
            area_counts = self._count_files(X_path)

            for root, dirs, files in os.walk(BASE_PATH):
                for file in sorted(files):
                        if file.endswith(ending):
                            path_split = root.split(os.sep)
                            area_name = os.sep.join(path_split[-3:-1]) #domain + area name - should be unique
                            count = area_counts[area_name]
                            for i in range(count):
                                paths.append(os.path.join(root, file))             
        return paths
    
    def _count_files(self, base_path):
        #This function counts files in subdirectories
        aerial_counts = {}
        
        for root, dirs, files in os.walk(base_path):
            if len(files) > 0:
                path_sections = root.split(os.sep)  # Split the path using the separator
                area_name = os.sep.join(path_sections[-3:-1])  # Join 
        
                aerial_counts[area_name] = len(files)
        return aerial_counts
    
    def prepare_dataset(self, part):
        X_BASE_PATH = os.path.join(self.config['path'], self.config['X_path'] + '_' + 'train')
        Y_BASE_PATH = os.path.join(self.config['path'], self.config['Y_path'] + '_' + 'train')
        SENTI_BASE_PATH = os.path.join(self.config['path'], self.config['senti_path'] + '_' + 'train')

        X_tif_paths = self._read_paths(X_BASE_PATH, '.tif')
        Y_tif_paths = self._read_paths(Y_BASE_PATH, '.tif')
        senti_data_paths = self._read_paths(SENTI_BASE_PATH, "data.npy", X_BASE_PATH) # all aerial images within the same area have the same 
        senti_mask_paths = self._read_paths(SENTI_BASE_PATH, "masks.npy", X_BASE_PATH)# sentinel image so redundant to store one for each 
        senti_dates_paths = self._read_paths(SENTI_BASE_PATH, "products.txt", X_BASE_PATH) 

        combined = list(zip(X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths))
        random.shuffle(combined)
        X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths = zip(*combined)
        X_tif_paths, Y_tif_paths, senti_data_paths, senti_mask_paths, senti_dates_paths = list(X_tif_paths), list(Y_tif_paths), list(senti_data_paths), list(senti_mask_paths), list(senti_dates_paths)
        assert len(X_tif_paths) == len(Y_tif_paths) == len(senti_data_paths) == len(senti_mask_paths) == len(senti_dates_paths)

        val_set_paths = ["D004", "D014", "D029", "D031", "D058", "D066", "D067", "D077"]
        if part == 'val':
            X_tif_paths = [path for path in X_tif_paths if any(s in path for s in val_set_paths)]
            Y_tif_paths = [path for path in Y_tif_paths if any(s in path for s in val_set_paths)]
            senti_data_paths = [path for path in senti_data_paths if any(s in path for s in val_set_paths)]
            senti_mask_paths = [path for path in senti_mask_paths if any(s in path for s in val_set_paths)]
            senti_dates_paths = [path for path in senti_dates_paths if any(s in path for s in val_set_paths)] 
        elif part == 'train':
            X_tif_paths = [path for path in X_tif_paths if not any(s in path for s in val_set_paths)]
            Y_tif_paths = [path for path in Y_tif_paths if not any(s in path for s in val_set_paths)]
            senti_data_paths = [path for path in senti_data_paths if not any(s in path for s in val_set_paths)]
            senti_mask_paths = [path for path in senti_mask_paths if not any(s in path for s in val_set_paths)]
            senti_dates_paths = [path for path in senti_dates_paths if not any(s in path for s in val_set_paths)]
        if part == 'train' or part == 'val':
            data_stop_point = math.floor(len(X_tif_paths) * (self.config['dataset_size']))
            X_tif_paths = X_tif_paths[:data_stop_point]
            Y_tif_paths = Y_tif_paths[:data_stop_point]
            senti_data_paths = senti_data_paths[0:data_stop_point]
            senti_mask_paths = senti_mask_paths[0:data_stop_point]
            senti_dates_paths= senti_dates_paths[0:data_stop_point]

        # Save the paths to files
        with open(f'../datasets/paths/X_paths_{part}_5.txt', 'w') as f:
            for path in X_tif_paths:
                f.write("%s\n" % path)
        with open(f'../datasets/paths/Y_paths_{part}_5.txt', 'w') as f:
            for path in Y_tif_paths:
                f.write("%s\n" % path)
        with open(f'../datasets/paths/senti_data_paths_{part}_5.txt', 'w') as f:
            for path in senti_data_paths:
                f.write("%s\n" % path)        
        with open(f'../datasets/paths/senti_mask_paths_{part}_5.txt', 'w') as f:
            for path in senti_mask_paths:
                f.write("%s\n" % path)  
        with open(f'../datasets/paths/senti_dates_paths_{part}_5.txt', 'w') as f:
            for path in senti_dates_paths:
                f.write("%s\n" % path)              
# Example usage
config = {
    'path': '../datasets/flair',
    'X_path': 'flair_aerial',
    'Y_path': 'flair_labels',
    'dataset_size': 0.05,  # Use 1.0 for full dataset, less for a fraction
    'senti_path': 'flair_sen'
}
random.seed(42)
preparer = DatasetPreparer(config)
preparer.prepare_dataset('train')
preparer.prepare_dataset('val')