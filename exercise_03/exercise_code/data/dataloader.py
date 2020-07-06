"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    @staticmethod       
    def combine_batch_dicts(batch):
        batch_dict = {}
        for data_dict in batch:
            for key, value in data_dict.items():
                if key not in batch_dict:
                    batch_dict[key] = []
                batch_dict[key].append(value)
        return batch_dict
    
    @staticmethod 
    def batch_to_numpy(batch):
        numpy_batch = {}
        for key, value in batch.items():
            numpy_batch[key] = np.array(value)
        return numpy_batch
            
    def __iter__(self):
        
        if self.shuffle is True:
            indices = np.random.permutation(len(self.dataset)) # define indices as iterator
            index_iterator = iter(indices) 
        else:
            indices = range(len(self.dataset))
            index_iterator = iter(indices)
            
        batch = []
        for index in index_iterator:  
            batch.append(self.dataset[index])
            if index == indices[-1] and self.drop_last is False:
                    yield self.batch_to_numpy(self.combine_batch_dicts(batch)) 
            if len(batch) == self.batch_size:
                yield self.batch_to_numpy(self.combine_batch_dicts(batch))
                batch = []

                
    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset  #
        ########################################################################

        if self.drop_last is not False:
            length = int(len(self.dataset)/self.batch_size)
        else:
            length = int(len(self.dataset)/self.batch_size) + 1
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
